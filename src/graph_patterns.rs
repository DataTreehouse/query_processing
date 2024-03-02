use crate::errors::QueryProcessingError;
use log::warn;
use oxrdf::Variable;
use polars::datatypes::{CategoricalOrdering, DataType};
use polars::frame::UniqueKeepStrategy;
use polars::prelude::{col, concat_lf_diagonal, lit, Expr, JoinArgs, JoinType, UnionArgs};
use representation::multitype::{convert_lf_col_to_multitype, create_join_compatible_solution_mappings, explode_multicols, implode_multicolumns};
use representation::query_context::{Context};
use representation::multitype::join_workaround;
use representation::solution_mapping::{is_string_col, SolutionMappings};
use representation::{BaseRDFNodeType, RDFNodeType};
use std::collections::{HashMap, HashSet};
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
        .drop([&expression_context.as_str()]);
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
    solution_mappings: SolutionMappings,
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
        mappings = mappings.drop([dummy_varname]);
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
                    right_mappings.with_column(col(&c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)));
                left_mappings =
                    left_mappings.with_column(col(&c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)));
            }
        }

        left_mappings = join_workaround(
            left_mappings,
            &left_datatypes,
            right_mappings,
            &right_datatypes,
            JoinType::Inner,
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
                    right_mappings.with_column(col(&c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)).alias(&c));
                left_mappings =
                    left_mappings.with_column(col(&c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)).alias(&c));
            }
        }
        left_mappings = join_workaround(
            left_mappings,
            &left_datatypes,
            right_mappings,
            &right_datatypes,
            JoinType::Left
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
                    right_mappings.with_column(col(c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)));
                left_solution_mappings.mappings = left_solution_mappings
                    .mappings
                    .with_column(col(c).cast(DataType::Categorical(None, CategoricalOrdering::Physical)));
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
    mappings = mappings.drop(
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
        mappings: mut left_mappings,
        rdf_node_types: left_datatypes,
    } = left_solution_mappings;
    let SolutionMappings {
        mappings: mut right_mappings,
        rdf_node_types: right_datatypes,
    } = right_solution_mappings;
    let mut updated_types = HashMap::new();
    let mut left_new_multitypes = HashMap::new();
    let mut right_new_multitypes = HashMap::new();
    for (left_col, left_type) in &left_datatypes {
        if let Some(right_type) = right_datatypes.get(left_col) {
            if left_type != right_type {
                if let RDFNodeType::MultiType(left_types) = left_type {
                    let mut left_set:HashSet<_> = left_types.iter().collect();
                    if let RDFNodeType::MultiType(right_types) = right_type {
                        let right_set:HashSet<_> = right_types.iter().collect();
                        let mut union:Vec<_> = left_set.union(&right_set).into_iter().map(|x|(*x).clone()).collect();
                        union.sort();
                        updated_types.insert(left_col.clone(), RDFNodeType::MultiType(union));
                    } else { //Right not multi
                        let base_right = BaseRDFNodeType::from_rdf_node_type(right_type);
                        left_set.insert(&base_right);
                        let mut new_types: Vec<_> = left_set.into_iter().map(|x|x.clone()).collect();
                        new_types.sort();
                        let new_type = RDFNodeType::MultiType(new_types);
                        right_mappings = convert_lf_col_to_multitype(right_mappings, left_col, right_type);
                        //Update the current multitype of right_mappings
                        right_new_multitypes.insert(left_col.clone(), RDFNodeType::MultiType(vec![base_right]));
                        updated_types.insert(left_col.clone(), new_type);
                    }
                } else { //Left not multi
                    if let RDFNodeType::MultiType(right_types) = right_type {
                        let mut right_set:HashSet<_> = right_types.iter().collect();
                        let base_left = BaseRDFNodeType::from_rdf_node_type(left_type);
                        right_set.insert(&base_left);
                        let mut new_types: Vec<_> = right_set.into_iter().map(|x|x.clone()).collect();
                        new_types.sort();
                        let new_type = RDFNodeType::MultiType(new_types);
                        left_mappings = convert_lf_col_to_multitype(left_mappings, left_col, left_type);
                        //Update the current multitype of left_mappings
                        left_new_multitypes.insert(left_col.clone(), RDFNodeType::MultiType(vec![base_left]));
                        updated_types.insert(left_col.clone(), new_type);
                    } else { //Both not multi
                        let base_left = BaseRDFNodeType::from_rdf_node_type(left_type);
                        let base_right = BaseRDFNodeType::from_rdf_node_type(right_type);
                        let mut new_types = vec![base_left.clone(), base_right.clone()];
                        new_types.sort();
                        let new_type = RDFNodeType::MultiType(new_types);
                        left_mappings = convert_lf_col_to_multitype(left_mappings, left_col, left_type);
                        right_mappings = convert_lf_col_to_multitype(right_mappings, left_col, right_type);

                        left_new_multitypes.insert(left_col.clone(), RDFNodeType::MultiType(vec![base_left]));
                        right_new_multitypes.insert(left_col.clone(), RDFNodeType::MultiType(vec![base_right]));

                        updated_types.insert(left_col.to_string(), new_type);
                    }
                }
            }
        }
    }
    for (c, t) in &left_datatypes {
        if matches!(t, &RDFNodeType::MultiType(..)) {
            left_new_multitypes.insert(c.clone(), t.clone());
        }
    }

    for (c, t) in &right_datatypes {
        if matches!(t, &RDFNodeType::MultiType(..)) {
            right_new_multitypes.insert(c.clone(), t.clone());
        }
    }

    let (left_mappings, mut left_exploded_map) = explode_multicols(left_mappings, &left_new_multitypes);
    let (right_mappings, right_exploded_map) = explode_multicols(right_mappings, &right_new_multitypes);
    for (c, (right_inner_columns, prefixed_right_inner_columns)) in right_exploded_map {
        if let Some((left_inner_columns, prefixed_left_inner_columns)) = left_exploded_map.get_mut(c) {
            for (r,pr) in right_inner_columns.into_iter().zip(prefixed_right_inner_columns.into_iter()) {
                if !left_inner_columns.contains(&r) {
                    left_inner_columns.push(r);
                    prefixed_left_inner_columns.push(pr);
                }
            }
        }
    }

    let mut output_mappings =
        concat_lf_diagonal(vec![left_mappings, right_mappings], UnionArgs::default())
            .expect("Concat problem");
    output_mappings = implode_multicolumns(output_mappings, left_exploded_map);
    let mut out_map = HashMap::new();
    for (k,v) in left_datatypes.into_iter().chain(right_datatypes.into_iter()) {
        if !out_map.contains_key(&k) {
            if let Some(v) = updated_types.remove(&k) {
                out_map.insert(k, v);
            } else {
                out_map.insert(k, v);
            }
        }
    }

    Ok(SolutionMappings::new(output_mappings, out_map))
}
