use feathertalk_face::{FaceError, ImageSize, generate_anchor_centers, resize_with_padding};

#[test]
fn computes_square_portrait_and_landscape_transforms() {
    let square = resize_with_padding(ImageSize {
        width: 640,
        height: 640,
    })
    .unwrap();
    assert_eq!(
        (
            square.new_width,
            square.new_height,
            square.pad_x,
            square.pad_y
        ),
        (640, 640, 0, 0)
    );

    let portrait = resize_with_padding(ImageSize {
        width: 360,
        height: 640,
    })
    .unwrap();
    assert_eq!((portrait.new_width, portrait.new_height), (360, 640));
    assert_eq!((portrait.pad_x, portrait.pad_y), (140, 0));

    let landscape = resize_with_padding(ImageSize {
        width: 640,
        height: 359,
    })
    .unwrap();
    assert_eq!((landscape.new_width, landscape.new_height), (640, 360));
    assert_eq!((landscape.pad_x, landscape.pad_y), (0, 140));
}

#[test]
fn rejects_zero_image_dimensions() {
    assert!(matches!(
        resize_with_padding(ImageSize {
            width: 0,
            height: 1
        }),
        Err(FaceError::InvalidImageSize)
    ));
}

#[test]
fn generates_schema_one_anchor_counts_and_order() {
    let model = ImageSize {
        width: 640,
        height: 640,
    };
    for (stride, expected) in [(8, 12_800), (16, 3_200), (32, 800)] {
        let anchors = generate_anchor_centers(model, stride, 2).unwrap();
        assert_eq!(anchors.len(), expected);
        assert_eq!(anchors[0], [0.0, 0.0]);
        assert_eq!(anchors[1], [0.0, 0.0]);
        assert_eq!(anchors[2], [stride as f32, 0.0]);
        assert_eq!(
            *anchors.last().unwrap(),
            [640.0 - stride as f32, 640.0 - stride as f32]
        );
    }
}

#[test]
fn rejects_invalid_anchor_configuration() {
    assert!(
        generate_anchor_centers(
            ImageSize {
                width: 320,
                height: 640
            },
            8,
            2
        )
        .is_err()
    );
    assert!(
        generate_anchor_centers(
            ImageSize {
                width: 640,
                height: 640
            },
            4,
            2
        )
        .is_err()
    );
    assert!(
        generate_anchor_centers(
            ImageSize {
                width: 640,
                height: 640
            },
            8,
            1
        )
        .is_err()
    );
}
