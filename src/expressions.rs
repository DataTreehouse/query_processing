use crate::errors::QueryProcessingError;
use oxrdf::vocab::xsd;
use oxrdf::{Literal, NamedNode, Variable};
use polars::datatypes::{DataType, TimeUnit};
use polars::frame::UniqueKeepStrategy;
use polars::prelude::{
    coalesce, col, concat_str, is_in, lit, Expr, IntoLazy, LazyFrame, LiteralValue,
    Operator, Series,
};
use representation::query_context::Context;
use representation::solution_mapping::SolutionMappings;
use representation::sparql_to_polars::{
    sparql_literal_to_polars_literal_value, sparql_named_node_to_polars_literal_value,
};
use representation::RDFNodeType;
use spargebra::algebra::{Expression, Function};
use std::collections::HashMap;
use std::ops::{Div, Mul};
use crate::constants::{DATETIME_AS_NANOS, DATETIME_AS_SECONDS, FLOOR_DATETIME_TO_SECONDS_INTERVAL, MODULUS, NANOS_AS_DATETIME, SECONDS_AS_DATETIME};

pub fn named_node(
    mut solution_mappings: SolutionMappings,
    nn: &NamedNode,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings.mappings.with_column(
        Expr::Literal(sparql_named_node_to_polars_literal_value(nn)).alias(context.as_str()),
    );
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), RDFNodeType::IRI);
    Ok(solution_mappings)
}

pub fn literal(
    mut solution_mappings: SolutionMappings,
    lit: &Literal,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings.mappings.with_column(
        Expr::Literal(sparql_literal_to_polars_literal_value(lit)).alias(context.as_str()),
    );
    solution_mappings.rdf_node_types.insert(
        context.as_str().to_string(),
        RDFNodeType::Literal(lit.datatype().into_owned()),
    );
    Ok(solution_mappings)
}

pub fn variable(
    mut solution_mappings: SolutionMappings,
    v: &Variable,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    if !solution_mappings.rdf_node_types.contains_key(v.as_str()) {
        return Err(QueryProcessingError::VariableNotFound(
            v.as_str().to_string(),
            context.as_str().to_string(),
        ));
    }
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(col(v.as_str()).alias(context.as_str()));
    let existing_type = solution_mappings.rdf_node_types.get(v.as_str()).unwrap();
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), existing_type.clone());
    Ok(solution_mappings)
}

pub fn binary_expression(
    mut solution_mappings: SolutionMappings,
    op: Operator,
    left_context: &Context,
    right_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(
            (Expr::BinaryExpr {
                left: Box::new(col(left_context.as_str())),
                op,
                right: Box::new(col(right_context.as_str())),
            })
            .alias(context.as_str()),
        )
        .drop_columns([left_context.as_str(), right_context.as_str()]);
    let t = match op {
        Operator::LtEq
        | Operator::GtEq
        | Operator::Gt
        | Operator::Lt
        | Operator::And
        | Operator::Eq
        | Operator::Or => RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
        Operator::Plus | Operator::Minus | Operator::Multiply | Operator::Divide => {
            let left_type = solution_mappings
                .rdf_node_types
                .get(left_context.as_str())
                .unwrap();
            let right_type = solution_mappings
                .rdf_node_types
                .get(right_context.as_str())
                .unwrap();
            let div_int = if op == Operator::Divide {
                if let RDFNodeType::Literal(right_lit) = right_type {
                    matches!(
                        right_lit.as_ref(),
                        xsd::INT
                            | xsd::LONG
                            | xsd::INTEGER
                            | xsd::BYTE
                            | xsd::SHORT
                            | xsd::UNSIGNED_INT
                            | xsd::UNSIGNED_LONG
                            | xsd::UNSIGNED_BYTE
                            | xsd::UNSIGNED_SHORT
                    )
                } else {
                    false
                }
            } else {
                false
            };

            if div_int {
                RDFNodeType::Literal(xsd::DOUBLE.into_owned())
                //TODO: Fix,
            } else {
                left_type.clone()
            }
        }
        _ => {
            panic!()
        }
    };

    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), t);
    Ok(solution_mappings)
}

pub fn unary_plus(
    mut solution_mappings: SolutionMappings,
    plus_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(col(plus_context.as_str()).alias(context.as_str()))
        .drop_columns([&plus_context.as_str()]);
    let existing_type = solution_mappings
        .rdf_node_types
        .get(plus_context.as_str())
        .unwrap();
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), existing_type.clone());
    Ok(solution_mappings)
}

pub fn unary_minus(
    mut solution_mappings: SolutionMappings,
    minus_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings //TODO: This is probably wrong
        .mappings
        .with_column(
            (Expr::BinaryExpr {
                left: Box::new(Expr::Literal(LiteralValue::Int32(0))),
                op: Operator::Minus,
                right: Box::new(col(minus_context.as_str())),
            })
            .alias(context.as_str()),
        )
        .drop_columns([&minus_context.as_str()]);
    let existing_type = solution_mappings
        .rdf_node_types
        .get(minus_context.as_str())
        .unwrap();
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), existing_type.clone());
    Ok(solution_mappings)
}

pub fn not_expression(
    mut solution_mappings: SolutionMappings,
    not_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(col(not_context.as_str()).not().alias(context.as_str()))
        .drop_columns([&not_context.as_str()]);
    solution_mappings.rdf_node_types.insert(
        context.as_str().to_string(),
        RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
    );
    Ok(solution_mappings)
}

pub fn bound(
    mut solution_mappings: SolutionMappings,
    v: &Variable,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(col(v.as_str()).is_null().alias(context.as_str()));
    solution_mappings.rdf_node_types.insert(
        context.as_str().to_string(),
        RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
    );
    Ok(solution_mappings)
}

pub fn if_expression(
    mut solution_mappings: SolutionMappings,
    left_context: &Context,
    middle_context: &Context,
    right_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(
            (Expr::Ternary {
                predicate: Box::new(col(left_context.as_str())),
                truthy: Box::new(col(middle_context.as_str())),
                falsy: Box::new(col(right_context.as_str())),
            })
            .alias(context.as_str()),
        )
        .drop_columns([
            left_context.as_str(),
            middle_context.as_str(),
            right_context.as_str(),
        ]);
    //Todo: generalize..
    let existing_type = solution_mappings
        .rdf_node_types
        .get(middle_context.as_str())
        .unwrap();
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), existing_type.clone());
    Ok(solution_mappings)
}

pub fn coalesce_expression(
    mut solution_mappings: SolutionMappings,
    inner_contexts: Vec<Context>,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    let mut coal_exprs = vec![];
    for c in &inner_contexts {
        coal_exprs.push(col(c.as_str()));
    }

    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(coalesce(coal_exprs.as_slice()).alias(context.as_str()))
        .drop_columns(
            inner_contexts
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<&str>>(),
        );
    //TODO: generalize
    let existing_type = solution_mappings
        .rdf_node_types
        .get(inner_contexts.first().unwrap().as_str())
        .unwrap();
    solution_mappings
        .rdf_node_types
        .insert(context.as_str().to_string(), existing_type.clone());
    Ok(solution_mappings)
}

pub fn exists(
    solution_mappings: SolutionMappings,
    exists_lf: LazyFrame,
    exists_context: &Context,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    let SolutionMappings {
        mappings,
        rdf_node_types,
    } = solution_mappings;
    let mut df = mappings.collect().unwrap();
    let exists_df = exists_lf
        .select([col(exists_context.as_str())])
        .unique(None, UniqueKeepStrategy::First)
        .collect()
        .expect("Collect lazy exists error");
    let mut ser = Series::from(
        is_in(
            //TODO: Fix - this can now work in lazy
            df.column(exists_context.as_str()).unwrap(),
            exists_df.column(exists_context.as_str()).unwrap(),
        )
        .unwrap(),
    );
    ser.rename(context.as_str());
    df.with_column(ser).unwrap();
    df = df.drop(exists_context.as_str()).unwrap();
    Ok(SolutionMappings::new(df.lazy(), rdf_node_types))
}

pub fn func_expression(
    mut solution_mappings: SolutionMappings,
    func: &Function,
    args: &Vec<Expression>,
    args_contexts: HashMap<usize, Context>,
    context: &Context,
) -> Result<SolutionMappings, QueryProcessingError> {
    match func {
        Function::Year => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .year()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Month => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .month()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Day => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .day()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Hours => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .hour()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Minutes => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .minute()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Seconds => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings.mappings.with_column(
                col(first_context.as_str())
                    .dt()
                    .second()
                    .alias(context.as_str()),
            );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::UNSIGNED_INT.into_owned()),
            );
        }
        Function::Abs => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings
                .mappings
                .with_column(col(first_context.as_str()).abs().alias(context.as_str()));
            let existing_type = solution_mappings
                .rdf_node_types
                .get(first_context.as_str())
                .unwrap();
            solution_mappings
                .rdf_node_types
                .insert(context.as_str().to_string(), existing_type.clone());
        }
        Function::Ceil => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings
                .mappings
                .with_column(col(first_context.as_str()).ceil().alias(context.as_str()));
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::INTEGER.into_owned()),
            );
        }
        Function::Floor => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings
                .mappings
                .with_column(col(first_context.as_str()).floor().alias(context.as_str()));
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::INTEGER.into_owned()),
            );
        }
        Function::Concat => {
            assert!(args.len() > 1);
            let SolutionMappings {
                mappings,
                rdf_node_types: datatypes,
            } = solution_mappings;
            let cols: Vec<_> = (0..args.len())
                .map(|i| col(args_contexts.get(&i).unwrap().as_str()))
                .collect();
            let new_mappings = mappings.with_column(concat_str(cols, "").alias(context.as_str()));
            solution_mappings = SolutionMappings::new(new_mappings, datatypes);
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::STRING.into_owned()),
            );
        }
        Function::Round => {
            assert_eq!(args.len(), 1);
            let first_context = args_contexts.get(&0).unwrap();
            solution_mappings.mappings = solution_mappings
                .mappings
                .with_column(col(first_context.as_str()).round(0).alias(context.as_str()));
            let existing_type = solution_mappings
                .rdf_node_types
                .get(first_context.as_str())
                .unwrap();
            solution_mappings
                .rdf_node_types
                .insert(context.as_str().to_string(), existing_type.clone());
        }
        Function::Regex => {
            if args.len() != 2 {
                todo!("Unsupported amount of regex args {:?}", args);
            } else {
                let first_context = args_contexts.get(&0).unwrap();
                if let Expression::Literal(l) = args.get(1).unwrap() {
                    solution_mappings.mappings = solution_mappings.mappings.with_column(
                        col(first_context.as_str())
                            .str()
                            .contains(lit(l.value()), false)
                            .alias(context.as_str()),
                    );
                    solution_mappings.rdf_node_types.insert(
                        context.as_str().to_string(),
                        RDFNodeType::Literal(xsd::STRING.into_owned()),
                    );
                }
            }
        }
        Function::Custom(nn) => {
            let iri = nn.as_str();
            if iri == xsd::INTEGER.as_str() {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings = solution_mappings.mappings.with_column(
                    col(first_context.as_str())
                        .cast(DataType::Int64)
                        .alias(context.as_str()),
                );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::INTEGER.into_owned()),
                );
            } else if iri == xsd::STRING.as_str() {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings = solution_mappings.mappings.with_column(
                    col(first_context.as_str())
                        .cast(DataType::Utf8)
                        .alias(context.as_str()),
                );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::STRING.into_owned()),
                );
            } else if iri == DATETIME_AS_NANOS {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        col(&first_context.as_str())
                            .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
                            .cast(DataType::UInt64)
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::INTEGER.into_owned()),
                );
            } else if iri == DATETIME_AS_SECONDS {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        col(&first_context.as_str())
                            .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                            .cast(DataType::UInt64)
                            .div(lit(1000))
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::INTEGER.into_owned()),
                );
            } else if iri == NANOS_AS_DATETIME {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        col(&first_context.as_str())
                            .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::DATE_TIME.into_owned()),
                );
            } else if iri == SECONDS_AS_DATETIME {
                assert_eq!(args.len(), 1);
                let first_context = args_contexts.get(&0).unwrap();
                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        col(&first_context.as_str())
                            .mul(Expr::Literal(LiteralValue::UInt64(1000)))
                            .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::DATE_TIME.into_owned()),
                );
            } else if iri == MODULUS {
                assert_eq!(args.len(), 2);
                let first_context = args_contexts.get(&0).unwrap();
                let second_context = args_contexts.get(&1).unwrap();

                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        (col(&first_context.as_str()) % col(&second_context.as_str()))
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::INTEGER.into_owned()),
                );
            } else if iri == FLOOR_DATETIME_TO_SECONDS_INTERVAL {
                assert_eq!(args.len(), 2);
                let first_context = args_contexts.get(&0).unwrap();
                let second_context = args_contexts.get(&1).unwrap();

                let first_as_seconds = col(&first_context.as_str())
                    .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                    .cast(DataType::UInt64)
                    .div(lit(1000));

                solution_mappings.mappings =
                    solution_mappings.mappings.with_column(
                        ((first_as_seconds.clone()
                            - (first_as_seconds % col(&second_context.as_str())))
                            .mul(Expr::Literal(LiteralValue::UInt64(1000)))
                            .cast(DataType::Datetime(TimeUnit::Milliseconds, None)))
                            .alias(context.as_str()),
                    );
                solution_mappings.rdf_node_types.insert(
                    context.as_str().to_string(),
                    RDFNodeType::Literal(xsd::DATE_TIME.into_owned()),
                );
            } else {
                todo!("{:?}", nn)
            }
        }
        Function::Contains => {
            assert_eq!(args.len(), 2);
            let first_context = args_contexts.get(&0).unwrap();
            let second_context = args_contexts.get(&1).unwrap();

            solution_mappings.mappings =
                solution_mappings.mappings.with_column(
                    (col(&first_context.as_str()).str().contains_literal(col(&second_context.as_str())))
                        .alias(context.as_str()),
                );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
            );
        }
        Function::StrStarts => {
            assert_eq!(args.len(), 2);
            let first_context = args_contexts.get(&0).unwrap();
            let second_context = args_contexts.get(&1).unwrap();

            solution_mappings.mappings =
                solution_mappings.mappings.with_column(
                    (col(&first_context.as_str()).str().starts_with(col(&second_context.as_str())))
                        .alias(context.as_str()),
                );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
            );
        }
        Function::StrEnds => {
            assert_eq!(args.len(), 2);
            let first_context = args_contexts.get(&0).unwrap();
            let second_context = args_contexts.get(&1).unwrap();

            solution_mappings.mappings =
                solution_mappings.mappings.with_column(
                    (col(&first_context.as_str()).str().ends_with(col(&second_context.as_str())))
                        .alias(context.as_str()),
                );
            solution_mappings.rdf_node_types.insert(
                context.as_str().to_string(),
                RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
            );
        }
        _ => {todo!("{}", func)}
    }
    solution_mappings.mappings = solution_mappings.mappings.drop_columns(
        args_contexts
            .values()
            .map(|x| x.as_str())
            .collect::<Vec<&str>>(),
    );
    Ok(solution_mappings)
}

pub fn in_expression(mut solution_mappings: SolutionMappings, left_context:&Context, right_contexts:&Vec<Context>, context: &Context) -> Result<SolutionMappings, QueryProcessingError> {
    let mut expr = Expr::Literal(LiteralValue::Boolean(false));

    for right_context in right_contexts {
        expr = Expr::BinaryExpr {
            left: Box::new(expr),
            op: Operator::Or,
            right: Box::new(Expr::BinaryExpr {
                left: Box::new(col(left_context.as_str())),
                op: Operator::Eq,
                right: Box::new(col(right_context.as_str())),
            }),
        }
    }

    solution_mappings.mappings = solution_mappings
        .mappings
        .with_column(expr.alias(context.as_str()))
        .drop_columns([left_context.as_str()])
        .drop_columns(
            right_contexts
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<&str>>(),
        );
    solution_mappings.rdf_node_types.insert(
        context.as_str().to_string(),
        RDFNodeType::Literal(xsd::BOOLEAN.into_owned()),
    );
    Ok(solution_mappings)
}