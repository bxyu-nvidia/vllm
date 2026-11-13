// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use vllm_engine_core_client::protocol::dtype::TensorDtype;
use vllm_engine_core_client::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

use crate::error::{ApiError, server_error};

/// Match Python's base64-encoded NumPy array, preserving route order and dtype.
pub(super) fn encode(
    chunks: &[WireNdArray],
    dropped_tokens: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    prompt_start: usize,
) -> Result<Option<String>, ApiError> {
    let Some(first) = chunks.first() else {
        return Ok(None);
    };
    if first.shape.len() != 3
        || !matches!(
            first.dtype.scalar,
            TensorDtype::U8
                | TensorDtype::U16
                | TensorDtype::U32
                | TensorDtype::U64
                | TensorDtype::I8
                | TensorDtype::I16
                | TensorDtype::I32
                | TensorDtype::I64
        )
    {
        return Err(server_error!(
            "expert routes must be a rank-3 integer array"
        ));
    }
    let mut rows = 0usize;
    for chunk in chunks {
        if chunk.dtype != first.dtype
            || chunk.shape.len() != 3
            || chunk.shape[1..] != first.shape[1..]
        {
            return Err(server_error!("inconsistent expert-route chunks"));
        }
        let expected_bytes = chunk
            .shape
            .checked_numel()
            .and_then(|count| count.checked_mul(chunk.dtype.scalar.element_size()))
            .ok_or_else(|| server_error!("expert-route shape overflows"))?;
        match &chunk.data {
            WireArrayData::RawView(bytes) if bytes.len() == expected_bytes => {}
            _ => {
                return Err(server_error!(
                    "expert-route payload has invalid byte length or unresolved data"
                ));
            }
        }
        rows = rows
            .checked_add(chunk.shape[0])
            .ok_or_else(|| server_error!("expert-route rows overflow"))?;
    }
    let retained = rows
        .checked_sub(dropped_tokens)
        .ok_or_else(|| server_error!("stop-string truncation exceeds expert-route rows"))?;
    let expected = prompt_tokens
        .checked_add(generated_tokens)
        .and_then(|count| count.checked_sub(1))
        .and_then(|count| count.checked_sub(prompt_start))
        .ok_or_else(|| server_error!("invalid expert-route token boundary"))?;
    if retained != expected {
        return Err(server_error!(
            "expert-route row count {retained} does not match expected {expected}"
        ));
    }
    let mut shape = first.shape.clone();
    shape[0] = retained;
    let bytes_to_keep = shape
        .checked_numel()
        .and_then(|count| count.checked_mul(first.dtype.scalar.element_size()))
        .ok_or_else(|| server_error!("expert-route byte length overflows"))?;
    // NPY v1.0 uses a little-endian u16 header length and a padded Python dict.
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        first.dtype, shape[0], shape[1], shape[2],
    );
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len())
        .map_err(|_| server_error!("expert-route header is too long"))?;
    let mut npy = Vec::with_capacity(10 + header.len() + bytes_to_keep);
    npy.extend_from_slice(b"\x93NUMPY\x01\x00");
    npy.extend_from_slice(&header_len.to_le_bytes());
    npy.extend_from_slice(header.as_bytes());
    let mut remaining = bytes_to_keep;
    for chunk in chunks {
        let WireArrayData::RawView(bytes) = &chunk.data else {
            unreachable!()
        };
        let take = remaining.min(bytes.len());
        npy.extend_from_slice(&bytes[..take]);
        remaining -= take;
    }
    Ok(Some(STANDARD.encode(npy)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_keep_prefill_rows_and_trim_only_dropped_generation_tokens() {
        let chunks = vec![
            WireNdArray::from_i32(vec![3, 1, 2], vec![0, 1, 2, 3, 4, 5]).unwrap(),
            WireNdArray::from_i32(vec![2, 1, 2], vec![6, 7, 8, 9]).unwrap(),
        ];
        let encoded = encode(&chunks, 1, 4, 2, 1).unwrap().unwrap();
        let npy = STANDARD.decode(encoded).unwrap();
        assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
        let header_end = 10 + u16::from_le_bytes([npy[8], npy[9]]) as usize;
        let header = std::str::from_utf8(&npy[10..header_end]).unwrap();
        assert!(header.contains("'shape': (4, 1, 2)"));
        assert_eq!(header_end % 64, 0);
        assert_eq!(
            &npy[header_end..],
            (0i32..8).flat_map(i32::to_ne_bytes).collect::<Vec<_>>()
        );
        assert!(encode(&chunks, 0, 4, 2, 1).is_err());
    }

    #[test]
    fn invalid_routes_fail_instead_of_entering_training() {
        let valid = WireNdArray::from_i32(vec![2, 1, 2], vec![1, 2, 3, 4]).unwrap();
        let mut truncated = valid.clone();
        truncated.shape[0] = 3;
        assert!(encode(&[truncated], 0, 3, 1, 0).is_err());
        let different_layout = WireNdArray::from_i32(vec![1, 2, 1], vec![1, 2]).unwrap();
        assert!(encode(&[valid, different_layout], 0, 3, 1, 0).is_err());
        assert!(
            encode(
                &[WireNdArray::from_f32(vec![1, 1, 1], vec![0.0]).unwrap()],
                0,
                1,
                1,
                0
            )
            .is_err()
        );
        assert_eq!(encode(&[], 0, 3, 1, 0).unwrap(), None);
    }
}
