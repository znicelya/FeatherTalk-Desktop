use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::PreprocessError;

pub const PFLD_LANDMARK_COUNT: usize = 110;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Landmarks {
    points: Vec<Point>,
}

impl Landmarks {
    pub fn points(&self) -> &[Point] {
        &self.points
    }
    pub(crate) fn from_points(points: Vec<Point>) -> Self {
        Self { points }
    }
}

pub fn read_landmarks(path: &Path) -> Result<Landmarks, PreprocessError> {
    let bytes = fs::read(path).map_err(|source| PreprocessError::Io {
        operation: "read_landmarks",
        path: path.to_path_buf(),
        source,
    })?;
    let text = String::from_utf8(bytes).map_err(|_| PreprocessError::InvalidUtf8 {
        path: path.to_path_buf(),
    })?;
    let mut points = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<_> = line.split_whitespace().collect();
        if tokens.len() != 2 {
            return Err(PreprocessError::InvalidLine {
                path: path.to_path_buf(),
                line: line_number,
                message: "expected exactly two float tokens".into(),
            });
        }
        let x = tokens[0]
            .parse::<f32>()
            .map_err(|_| invalid_line(path, line_number, "invalid x coordinate"))?;
        let y = tokens[1]
            .parse::<f32>()
            .map_err(|_| invalid_line(path, line_number, "invalid y coordinate"))?;
        if !x.is_finite() || !y.is_finite() {
            return Err(PreprocessError::NonFiniteCoordinate {
                path: path.to_path_buf(),
                line: line_number,
            });
        }
        if x < 0.0 || y < 0.0 {
            return Err(PreprocessError::NegativeCoordinate {
                path: path.to_path_buf(),
                line: line_number,
            });
        }
        points.push(Point { x, y });
    }
    if points.len() != PFLD_LANDMARK_COUNT {
        return Err(PreprocessError::WrongLandmarkCount {
            path: path.to_path_buf(),
            expected: PFLD_LANDMARK_COUNT,
            actual: points.len(),
        });
    }
    Ok(Landmarks::from_points(points))
}

fn invalid_line(path: &Path, line: usize, message: &str) -> PreprocessError {
    PreprocessError::InvalidLine {
        path: PathBuf::from(path),
        line,
        message: message.into(),
    }
}
