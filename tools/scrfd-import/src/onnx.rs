use prost::Message;

use crate::{OnnxContract, ToolError};

#[derive(Clone, PartialEq, prost::Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, prost::Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<DimensionProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct DimensionProto {
    #[prost(int64, optional, tag = "1")]
    dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    dim_param: Option<String>,
}

pub(crate) fn parse_contract(bytes: &[u8]) -> Result<OnnxContract, ToolError> {
    let model =
        ModelProto::decode(bytes).map_err(|error| ToolError::OnnxDecode(error.to_string()))?;
    if model.opset_import.len() != 1 {
        return Err(contract_error("expected exactly one opset import"));
    }
    let opset = &model.opset_import[0];
    if !opset.domain.is_empty() || opset.version != 12 {
        return Err(contract_error(format!(
            "expected default-domain opset 12, got domain {:?} version {}",
            opset.domain, opset.version
        )));
    }
    let graph = model
        .graph
        .ok_or_else(|| contract_error("graph is missing"))?;
    if graph.input.len() != 1 {
        return Err(contract_error("expected exactly one graph input"));
    }
    let (input_name, input_elem_type, input_shape) = value_contract(&graph.input[0])?;
    if input_name != "images" || input_elem_type != 1 || input_shape != [1, 3, 640, 640] {
        return Err(contract_error(format!(
            "unexpected input {input_name:?} elem_type {input_elem_type} shape {input_shape:?}"
        )));
    }
    if graph.output.len() != 9 {
        return Err(contract_error("expected exactly nine graph outputs"));
    }

    let expected_shapes = [
        vec![1, 12_800, 1],
        vec![1, 3_200, 1],
        vec![1, 800, 1],
        vec![1, 12_800, 4],
        vec![1, 3_200, 4],
        vec![1, 800, 4],
        vec![1, 12_800, 10],
        vec![1, 3_200, 10],
        vec![1, 800, 10],
    ];
    let mut output_names = Vec::with_capacity(9);
    let mut output_shapes = Vec::with_capacity(9);
    for (index, (output, expected_shape)) in graph.output.iter().zip(expected_shapes).enumerate() {
        let (name, elem_type, shape) = value_contract(output)?;
        let expected_name = format!("out{index}");
        if name != expected_name || elem_type != 1 || shape != expected_shape {
            return Err(contract_error(format!(
                "unexpected output {index}: name {name:?}, elem_type {elem_type}, shape {shape:?}"
            )));
        }
        output_names.push(name);
        output_shapes.push(shape);
    }

    Ok(OnnxContract {
        opset: opset.version as u64,
        input_name,
        input_elem_type,
        input_shape,
        output_names: output_names.try_into().expect("exactly nine outputs"),
        output_shapes,
    })
}

fn value_contract(value: &ValueInfoProto) -> Result<(String, i32, Vec<usize>), ToolError> {
    if value.name.is_empty() {
        return Err(contract_error("value name is empty"));
    }
    let tensor = value
        .r#type
        .as_ref()
        .and_then(|value_type| value_type.tensor_type.as_ref())
        .ok_or_else(|| contract_error(format!("{} is not a tensor", value.name)))?;
    let shape = tensor
        .shape
        .as_ref()
        .ok_or_else(|| contract_error(format!("{} has no shape", value.name)))?;
    let mut dimensions = Vec::with_capacity(shape.dim.len());
    for dimension in &shape.dim {
        if dimension
            .dim_param
            .as_deref()
            .is_some_and(|value| !value.is_empty())
        {
            return Err(contract_error(format!(
                "{} has symbolic dimensions",
                value.name
            )));
        }
        let dimension_value = dimension
            .dim_value
            .ok_or_else(|| contract_error(format!("{} has an unknown dimension", value.name)))?;
        if dimension_value <= 0 {
            return Err(contract_error(format!(
                "{} has non-positive dimension {dimension_value}",
                value.name
            )));
        }
        dimensions.push(
            usize::try_from(dimension_value)
                .map_err(|_| contract_error(format!("{} dimension is too large", value.name)))?,
        );
    }
    Ok((value.name.clone(), tensor.elem_type, dimensions))
}

fn contract_error(message: impl Into<String>) -> ToolError {
    ToolError::SourceContract(message.into())
}
