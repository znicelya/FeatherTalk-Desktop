use crate::image::GrayImage;

/// `cv2.Laplacian(gray, cv2.CV_64F)` with the default `ksize = 1`.
///
/// The kernel is `[[0, 1, 0], [1, -4, 1], [0, 1, 0]]` with `BORDER_REFLECT_101`
/// borders. Every sample is an integer in `-1020..=1275`, so the f64 output is
/// exact and can be compared for equality against OpenCV.
///
/// The result is row-major with `width * height` entries.
pub fn laplacian_response(image: &GrayImage) -> Vec<f64> {
    let width = i64::from(image.width());
    let height = i64::from(image.height());
    let mut response = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let above = image.row(reflect_101(y - 1, height) as u32);
        let center = image.row(y as u32);
        let below = image.row(reflect_101(y + 1, height) as u32);
        for x in 0..width {
            let index = x as usize;
            let left = center[reflect_101(x - 1, width) as usize];
            let right = center[reflect_101(x + 1, width) as usize];
            response.push(
                f64::from(above[index]) + f64::from(left) - 4.0 * f64::from(center[index])
                    + f64::from(right)
                    + f64::from(below[index]),
            );
        }
    }
    response
}

/// Population variance (`ddof = 0`) of the Laplacian response.
///
/// This is the single input to the blur decision, so it mirrors
/// `cv2.Laplacian(..).var()` exactly rather than approximating it.
pub fn laplacian_variance(image: &GrayImage) -> f64 {
    let response = laplacian_response(image);
    debug_assert!(
        !response.is_empty(),
        "GrayImage always has a nonzero width and height"
    );
    let count = response.len() as f64;
    let mean = response.iter().sum::<f64>() / count;
    response
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count
}

/// OpenCV's `BORDER_REFLECT_101`: mirror without repeating the edge sample.
fn reflect_101(index: i64, size: i64) -> i64 {
    if size == 1 {
        return 0;
    }
    let period = 2 * (size - 1);
    let mut value = index.rem_euclid(period);
    if value >= size {
        value = period - value;
    }
    value
}
