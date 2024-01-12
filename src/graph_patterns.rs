use crate::errors::QueryProcessingError;
use log::warn;
use oxrdf::Variable;
use polars::datatypes::DataType;
use polars::export::ahash::HashSet;
use polars::frame::UniqueKeepStrategy;
use polars::prelude::{col, concat_lf_diagonal, lit, Expr, JoinArgs, JoinType, UnionArgs};
use representation::multitype::{
    create_compatible_solution_mappings, create_join_compatible_solution_mappings, join_workaround,
};
use representation::query_context::{Context, PathEntry};
use representation::solution_mapping::{is_string_col, SolutionMappings};
use representation::RDFNodeType;
use std::collections::HashMap;
use uuid::Uuid;

pub fn distinct(
    mut solution_mappings: SolutionMappings,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .unique_stable(None, UniqueKeepStrategy::First);
    Ok(solution_mappings)
}

pub fn extend(
    mut solution_mappings: SolutionMappings,
    expression_context: &Context,
    variable: &Variable,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .rename([expression_context.as_str()], [variable.as_str()]);
    let existing_rdf_node_type = solution_mappings
        .rdf_node_types
        .remove(expression_context.as_str())
        .unwrap();
    solution_mappings
        .rdf_node_types
        .insert(variable.as_str().to_string(), existing_rdf_node_type);
    Ok(solution_mappings)
}

pub fn filter(
    mut solution_mappings: SolutionMappings,
    expression_context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .filter(col(expression_context.as_str()))
        .drop_columns([&expression_context.as_str()]);
    Ok(solution_mappings)
}

pub fn prepare_group_by(
    mut solution_mappings: SolutionMappings,
    variables: &Vec<Variable>,
) -> (SolutionMappings, Vec<Expr>, Option<String>) {
    let by: Vec<Expr>;
    let dummy_varname = if variables.is_empty() {
        let dummy_varname = Uuid::new_v4().to_string();
        by = vec![col(&dummy_varname)];
        solution_mappings.mappings = solution_mappings
            .mappings
            .with_column(lit(true).alias(&dummy_varname));
        Some(dummy_varname)
    } else {
        by = variables.iter().map(|v| col(v.as_str())).collect();
        None
    };
    (solution_mappings, by, dummy_varname)
}

pub fn group_by(
    mut solution_mappings: SolutionMappings,
    aggregate_expressions: Vec<Expr>,
    by: Vec<Expr>,
    dummy_varname: Option<String>,
    new_rdf_node_types: HashMap<String, RDFNodeType>,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mut mappings,
        rdf_node_types: mut datatypes,
    } = solution_mappings;
    let grouped_mappings = mappings.group_by(by.as_slice());

    mappings = grouped_mappings.agg(aggregate_expressions.as_slice());
    for (k, v) in new_rdf_node_types {
        datatypes.insert(k, v);
    }
    if let Some(dummy_varname) = dummy_varname {
        mappings = mappings.drop_columns([dummy_varname]);
    }
    Ok(SolutionMappings::new(mappings, datatypes))
}

pub fn join(
    left_solution_mappings: SolutionMappings,
    right_solution_mappings: SolutionMappings,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mappings: right_mappings,
        rdf_node_types: right_datatypes,
    } = right_solution_mappings;

    let mut join_on: Vec<_> = {
        let right_column_set: HashSet<_> = right_datatypes.keys().collect();
        let left_column_set: HashSet<_> = left_solution_mappings.rdf_node_types.keys().collect();

        left_column_set
            .intersection(&right_column_set)
            .map(|x| (*x).clone())
            .collect()
    };
    join_on.sort();

    let join_on_cols: Vec<Expr> = join_on.iter().map(|x| col(x)).collect();

    let SolutionMappings {
        mappings: left_mappings,
        rdf_node_types: left_datatypes,
    } = left_solution_mappings;

    let (mut left_mappings, mut left_datatypes, mut right_mappings, right_datatypes) =
        create_join_compatible_solution_mappings(
            left_mappings,
            left_datatypes,
            right_mappings,
            right_datatypes,
            true,
        );

    if join_on.is_empty() {
        left_mappings = left_mappings.join(
            right_mappings,
            join_on_cols.as_slice(),
            join_on_cols.as_slice(),
            JoinArgs::new(JoinType::Cross),
        )
    } else {
        for c in join_on {
            let dt = right_datatypes.get(&c).unwrap();
            if is_string_col(dt) {
                right_mappings =
                    right_mappings.with_column(col(&c).cast(DataType::Categorical(None)));
                left_mappings =
                    left_mappings.with_column(col(&c).cast(DataType::Categorical(None)));
            }
        }

        left_mappings = join_workaround(
            left_mappings,
            &left_datatypes,
            right_mappings,
            &right_datatypes,
            JoinArgs::new(JoinType::Inner),
        );
    }

    for (k, v) in &right_datatypes {
        if !left_datatypes.contains_key(k) {
            left_datatypes.insert(k.clone(), v.clone());
        }
    }

    let left_solution_mappings = SolutionMappings {
        mappings: left_mappings,
        rdf_node_types: left_datatypes,
    };

    Ok(left_solution_mappings)
}

pub fn left_join(
    left_solution_mappings: SolutionMappings,
    right_solution_mappings: SolutionMappings,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mappings: right_mappings,
        rdf_node_types: right_datatypes,
    } = right_solution_mappings;

    let SolutionMappings {
        mappings: left_mappings,
        rdf_node_types: left_datatypes,
    } = left_solution_mappings;

    let mut join_on: Vec<_> = {
        let right_column_set: HashSet<_> = right_datatypes.keys().collect();
        let left_column_set: HashSet<_> = left_datatypes.keys().collect();

        left_column_set
            .intersection(&right_column_set)
            .map(|x| (*x).clone())
            .collect()
    };
    join_on.sort();

    let (mut left_mappings, mut left_datatypes, mut right_mappings, right_datatypes) =
        create_join_compatible_solution_mappings(
            left_mappings,
            left_datatypes,
            right_mappings,
            right_datatypes,
            false,
        );

    let join_on_cols: Vec<Expr> = join_on.iter().map(|x| col(x)).collect();

    if join_on.is_empty() {
        left_mappings = left_mappings.join(
            right_mappings,
            join_on_cols.as_slice(),
            join_on_cols.as_slice(),
            JoinArgs::new(JoinType::Cross),
        )
    } else {
        for c in join_on {
            let dt = right_datatypes.get(&c).unwrap();
            if is_string_col(dt) {
                right_mappings =
                    right_mappings.with_column(col(&c).cast(DataType::Categorical(None)).alias(&c));
                left_mappings =
                    left_mappings.with_column(col(&c).cast(DataType::Categorical(None)).alias(&c));
            }
        }
        left_mappings = join_workaround(
            left_mappings,
            &left_datatypes,
            right_mappings,
            &right_datatypes,
            JoinArgs {
                how: JoinType::Left,
                validation: Default::default(),
                suffix: None,
                slice: None,
            },
        );
    }

    for (k, v) in &right_datatypes {
        if !left_datatypes.contains_key(k) {
            left_datatypes.insert(k.clone(), v.clone());
        }
    }

    let left_solution_mappings = SolutionMappings {
        mappings: left_mappings,
        rdf_node_types: left_datatypes,
    };

    Ok(left_solution_mappings)
}

//TODO: Fix datatypes??!?!?
pub fn minus(
    mut left_solution_mappings: SolutionMappings,
    right_solution_mappings: SolutionMappings,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mappings: mut right_mappings,
        rdf_node_types: right_datatypes,
    } = right_solution_mappings;

    let right_column_set: HashSet<_> = right_datatypes.keys().collect();
    let left_column_set: HashSet<_> = left_solution_mappings.rdf_node_types.keys().collect();

    let mut join_on: Vec<_> = left_column_set
        .intersection(&right_column_set)
        .cloned()
        .collect();
    join_on.sort();

    if join_on.is_empty() {
        Ok(left_solution_mappings)
    } else {
        let join_on_cols: Vec<Expr> = join_on.iter().map(|x| col(x)).collect();
        for c in join_on {
            if is_string_col(left_solution_mappings.rdf_node_types.get(c).unwrap()) {
                right_mappings =
                    right_mappings.with_column(col(c).cast(DataType::Categorical(None)));
                left_solution_mappings.mappings = left_solution_mappings
                    .mappings
                    .with_column(col(c).cast(DataType::Categorical(None)));
            }
        }
        let all_false = [false].repeat(join_on_cols.len());
        right_mappings = right_mappings.sort_by_exprs(
            join_on_cols.as_slice(),
            all_false.as_slice(),
            false,
            false,
        );
        left_solution_mappings.mappings = left_solution_mappings.mappings.sort_by_exprs(
            join_on_cols.as_slice(),
            all_false.as_slice(),
            false,
            false,
        );
        left_solution_mappings.mappings = left_solution_mappings.mappings.join(
            right_mappings,
            join_on_cols.as_slice(),
            join_on_cols.as_slice(),
            JoinArgs::new(JoinType::Anti),
        );
        Ok(left_solution_mappings)
    }
}

pub fn order_by(
    solution_mappings: SolutionMappings,
    inner_contexts: &Vec<Context>,
    asc_ordering: Vec<bool>,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mut mappings,
        rdf_node_types: datatypes,
    } = solution_mappings;

    mappings = mappings.sort_by_exprs(
        inner_contexts
            .iter()
            .map(|c| col(c.as_str()))
            .collect::<Vec<Expr>>(),
        asc_ordering.iter().map(|asc| !asc).collect::<Vec<bool>>(),
        true,
        false,
    );
    mappings = mappings.drop_columns(
        inner_contexts
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<&str>>(),
    );

    Ok(SolutionMappings::new(mappings, datatypes))
}

pub fn project(
    solution_mappings: SolutionMappings,
    variables: &Vec<Variable>,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mut mappings,
        rdf_node_types: mut datatypes,
    } = solution_mappings;
    let cols: Vec<Expr> = variables.iter().map(|c| col(c.as_str())).collect();
    mappings = mappings.select(cols.as_slice());
    let mut new_datatypes = HashMap::new();
    for v in variables {
        if !datatypes.contains_key(v.as_str()) {
            warn!("Datatypes does not contain {}", v);
        } else {
            new_datatypes.insert(
                v.as_str().to_string(),
                datatypes.remove(v.as_str()).unwrap(),
            );
        }
    }
    Ok(SolutionMappings::new(mappings, new_datatypes))
}

pub fn union(
    left_solution_mappings: SolutionMappings,
    right_solution_mappings: SolutionMappings,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mappings: left_mappings,
        rdf_node_types: left_datatypes,
    } = left_solution_mappings;
    let SolutionMappings {
        mappings: right_mappings,
        rdf_node_types: right_datatypes,
    } = right_solution_mappings;
    let (left_mappings, mut left_datatypes, right_mappings, mut right_datatypes) =
        create_compatible_solution_mappings(
            left_mappings,
            left_datatypes,
            right_mappings,
            right_datatypes,
        );
    for (k, v) in right_datatypes.drain() {
        left_datatypes.entry(k).or_insert(v);
    }

    let output_mappings =
        concat_lf_diagonal(vec![left_mappings, right_mappings], UnionArgs::default())
            .expect("Concat problem");
    Ok(SolutionMappings::new(output_mappings, left_datatypes))
}
