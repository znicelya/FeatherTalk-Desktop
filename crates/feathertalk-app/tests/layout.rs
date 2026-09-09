#![cfg(feature = "ui-tests")]

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use feathertalk_app::{
    args::LaunchOptions, catalog, components::control_renderers, navigation::Page,
    pipeline::TaskUpdate, state::AppState, tasks::TaskCenter, theme, workbench::Workbench,
};
use feathertalk_client::CancelToken;
use feathertalk_domain::{Event, Progress, Request, TaskId, TaskKind, TaskStage};
use feathertalk_supervisor::supervisor::{SupervisedOutcome, SupervisionReport};
use gpui::{
    AnyElement, App, Bounds, Div, Entity, FocusHandle, Global, InteractiveElement, Modifiers,
    Pixels, Stateful, TestAppContext, VisualTestContext, Window, point, px, size,
};
use yororen_ui::{
    RendererContext,
    headless::{
        badge::BadgeProps, button::ButtonProps, number_input::NumberInputProps,
        progress::ProgressBarProps, text_input::TextInputState,
    },
    locale, markers, renderer,
    renderer::renderers::{
        BadgeRenderer, ButtonRenderer, NumberInputRenderer, ProgressBarRenderer,
    },
};

const TASK_ID: &str = "1756000000000-00000001";
const WINDOW_SIZES: [(f32, f32); 2] = [(960., 640.), (1280., 800.)];

struct MeasuredButtons(Arc<dyn ButtonRenderer>);

impl ButtonRenderer for MeasuredButtons {
    fn compose(&self, props: &ButtonProps, focus: &FocusHandle, cx: &App) -> Stateful<Div> {
        self.0
            .compose(props, focus, cx)
            .debug_selector(|| props.id.to_string())
    }
}

struct MeasuredBadges(Arc<dyn BadgeRenderer>);

struct ObservedNumberInputs(Arc<dyn NumberInputRenderer>);

struct BatchInput(Entity<TextInputState>);
impl Global for BatchInput {}

#[derive(Default)]
struct RenderedTrainingProgress(Mutex<Option<ProgressBarProps>>);
impl Global for RenderedTrainingProgress {}

struct ObservedProgress(Arc<dyn ProgressBarRenderer>);

impl ProgressBarRenderer for ObservedProgress {
    fn compose(&self, props: &ProgressBarProps, cx: &App) -> Div {
        if props.id.to_string() == format!("task-progress-{TASK_ID}") {
            *cx.global::<RenderedTrainingProgress>().0.lock().unwrap() = Some(props.clone());
        }
        self.0.compose(props, cx)
    }
}

impl NumberInputRenderer for ObservedNumberInputs {
    fn compose(&self, props: &NumberInputProps, cx: &mut App, window: &mut Window) -> AnyElement {
        let element = self.0.compose(props, cx, window);
        if props.id.to_string() == "training-batch-size" {
            let state = window.use_keyed_state(props.id.clone(), cx, |_, _| {
                panic!("the real number-input renderer must have created its state")
            });
            cx.set_global(BatchInput(state));
        }
        element
    }
}

impl BadgeRenderer for MeasuredBadges {
    fn compose(&self, props: &BadgeProps, cx: &App) -> Div {
        self.0
            .compose(props, cx)
            .debug_selector(|| props.id.to_string())
    }
}

fn install(cx: &mut App) {
    renderer::install_with(cx, theme::palette(false));
    control_renderers::install(cx);
    let buttons = cx
        .renderer_arc::<markers::Button, dyn ButtonRenderer>()
        .unwrap()
        .clone();
    cx.register_renderer_arc::<markers::Button, dyn ButtonRenderer>(Arc::new(MeasuredButtons(
        buttons,
    )));
    let badges = cx
        .renderer_arc::<markers::Badge, dyn BadgeRenderer>()
        .unwrap()
        .clone();
    cx.register_renderer_arc::<markers::Badge, dyn BadgeRenderer>(Arc::new(MeasuredBadges(badges)));
    let inputs = cx
        .renderer_arc::<markers::NumberInput, dyn NumberInputRenderer>()
        .unwrap()
        .clone();
    cx.register_renderer_arc::<markers::NumberInput, dyn NumberInputRenderer>(Arc::new(
        ObservedNumberInputs(inputs),
    ));
    let progress = cx
        .renderer_arc::<markers::ProgressBar, dyn ProgressBarRenderer>()
        .unwrap()
        .clone();
    cx.set_global(RenderedTrainingProgress::default());
    cx.register_renderer_arc::<markers::ProgressBar, dyn ProgressBarRenderer>(Arc::new(
        ObservedProgress(progress),
    ));
    yororen_ui::headless::text_input::init(cx);
    locale::install_with_translations(cx, catalog::LOCALE_TAG, catalog::translations().unwrap());
    let state = AppState::new(cx, LaunchOptions::default());
    cx.set_global(state);
}

// Deliver the same state transitions as the supervisor, without launching model
// jobs. These tests render the actual workbench pages and registered controls.
fn show_task(
    cx: &mut VisualTestContext,
    page: Page,
    kind: TaskKind,
    outcome: Option<SupervisedOutcome>,
) -> CancelToken {
    let cancel = CancelToken::new();
    cx.update(|window, cx| {
        let state = cx.global::<AppState>();
        let navigation = state.navigation.clone();
        let tasks = state.tasks.clone();
        navigation.update(cx, |navigation, cx| {
            navigation.select(page);
            cx.notify();
        });
        tasks.update(cx, |tasks, cx| {
            *tasks = TaskCenter::default();
            let id = TaskId::parse(TASK_ID).unwrap();
            if kind == TaskKind::Train {
                tasks.begin_training(id.clone(), 200, cancel.clone());
            } else {
                tasks.begin(id.clone(), kind, cancel.clone());
            }
            tasks.apply(TaskUpdate::Progress(Box::new(Event::new(
                id.clone(),
                "2026-09-08T00:00:00Z",
                TaskStage::Preparing,
            ))));
            if let Some(outcome) = outcome {
                tasks.apply(TaskUpdate::Finished(Box::new(SupervisionReport {
                    task_id: id,
                    attempts: 1,
                    outcome,
                    crash_logs: Vec::new(),
                    journal_errors: Vec::new(),
                })));
            }
            cx.notify();
        });
        window.refresh();
    });
    cx.run_until_parked();
    cancel
}

fn bounds(cx: &mut VisualTestContext, selector: &'static str) -> Bounds<Pixels> {
    cx.debug_bounds(selector)
        .unwrap_or_else(|| panic!("missing rendered element: {selector}"))
}

fn assert_inline_text(text: Bounds<Pixels>, minimum_width: f32, context: &str) {
    assert!(
        text.size.width >= px(minimum_width) && text.size.height <= px(26.),
        "{context}: short text should occupy a readable horizontal line, got {text:?}"
    );
}

fn assert_aligned(left: Bounds<Pixels>, right: Bounds<Pixels>, context: &str) {
    assert!(
        left.right() <= right.left() && f32::from(left.center().y - right.center().y).abs() <= 1.,
        "{context}: neighboring controls should share a row without overlap: {left:?}, {right:?}"
    );
}

#[gpui::test]
fn cancellation_hints_keep_their_width_and_align_with_buttons(cx: &mut TestAppContext) {
    cx.update(install);
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    for (width, height) in WINDOW_SIZES {
        cx.simulate_resize(size(px(width), px(height)));
        for (page, kind) in [
            (Page::Assets, TaskKind::NormalizeMedia),
            (Page::Assets, TaskKind::ExtractFrames),
            (Page::Assets, TaskKind::ExtractFeatures),
            (Page::Assets, TaskKind::ValidateProject),
            (Page::Training, TaskKind::Train),
            (Page::Generate, TaskKind::Render),
            (Page::Models, TaskKind::InspectModel),
            (Page::Tasks, TaskKind::Train),
        ] {
            let cancel = show_task(cx, page, kind, None);
            for requests in 0..=2 {
                let context =
                    format!("{width}x{height} {page:?} {kind:?}, stop requests={requests}");
                let row = bounds(cx, "task-cancel-row-1756000000000-00000001");
                let button = bounds(cx, "task-cancel-1756000000000-00000001");
                let hint = bounds(cx, "task-cancel-hint-1756000000000-00000001");
                assert_inline_text(hint, 90., &context);
                assert!(button.size.height <= px(40.), "{context}: {button:?}");
                assert_aligned(button, hint, &context);
                assert!(
                    hint.right() <= row.right() + px(1.),
                    "{context}: {row:?}, {hint:?}"
                );
                cancel.request();
                cx.update(|window, cx| {
                    let tasks = cx.global::<AppState>().tasks.clone();
                    tasks.update(cx, |_, cx| cx.notify());
                    window.refresh();
                });
                cx.run_until_parked();
            }
        }
    }
}

#[gpui::test]
fn latest_task_labels_align_with_completed_and_cancelled_badges(cx: &mut TestAppContext) {
    cx.update(install);
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    for (width, height) in WINDOW_SIZES {
        cx.simulate_resize(size(px(width), px(height)));
        for (page, kind, label, badge) in [
            (
                Page::Assets,
                TaskKind::ValidateProject,
                "assets-result-title",
                "assets-result-status-1756000000000-00000001",
            ),
            (
                Page::Training,
                TaskKind::Train,
                "training-latest-label",
                "training-latest-status",
            ),
        ] {
            for outcome in [
                SupervisedOutcome::Completed { result: None },
                SupervisedOutcome::Cancelled,
            ] {
                let context = format!("{width}x{height} {page:?} {outcome:?}");
                show_task(cx, page, kind, Some(outcome));
                let label = bounds(cx, label);
                let badge = bounds(cx, badge);
                assert_inline_text(label, 40., &context);
                assert_aligned(label, badge, &context);
            }
        }
    }
}

#[gpui::test]
fn short_preview_duration_is_a_horizontal_line(cx: &mut TestAppContext) {
    cx.update(install);
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    for (width, height) in WINDOW_SIZES {
        cx.simulate_resize(size(px(width), px(height)));
        for frames in [125., 250., 1000.] {
            cx.update(|window, cx| {
                let state = cx.global::<AppState>();
                let navigation = state.navigation.clone();
                let generate = state.generate.clone();
                navigation.update(cx, |navigation, cx| {
                    navigation.select(Page::Generate);
                    cx.notify();
                });
                generate.update(cx, |form, cx| {
                    form.set_preview(true);
                    form.set_preview_frames(frames);
                    cx.notify();
                });
                window.refresh();
            });
            cx.run_until_parked();
            assert_inline_text(
                bounds(cx, "generate-preview-estimate"),
                60.,
                &format!("{width}x{height}, preview frames={frames}"),
            );
        }
    }
}

#[gpui::test]
fn long_cancellation_hints_wrap_without_overflowing_the_action_row(cx: &mut TestAppContext) {
    cx.update(install);
    cx.update(|cx| {
        let mut translations = catalog::translations().unwrap();
        translations.merge(yororen_ui::i18n::parse_translation_value(serde_json::json!({
            "ui": { "tasks": { "cancel_hint": "停止后保留已经写出的结果，较长的说明应当在可用宽度内换行，而不是溢出容器或变成逐字竖排。".repeat(4) } }
        })).unwrap());
        locale::install_with_translations(cx, catalog::LOCALE_TAG, translations);
    });
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    // Also stress the row below the app's minimum window size: the hint should
    // move below the button as a whole when there is too little room beside it.
    for width in [960., 640.] {
        cx.simulate_resize(size(px(width), px(640.)));
        show_task(cx, Page::Assets, TaskKind::NormalizeMedia, None);
        let row = bounds(cx, "task-cancel-row-1756000000000-00000001");
        let hint = bounds(cx, "task-cancel-hint-1756000000000-00000001");
        let button = bounds(cx, "task-cancel-1756000000000-00000001");
        assert!(
            hint.size.width >= px(200.) && hint.right() <= row.right() + px(1.),
            "{row:?}, {hint:?}"
        );
        assert!(
            hint.size.height > px(26.) && hint.size.height <= px(180.),
            "{hint:?}"
        );
        assert!(hint.bottom() <= row.bottom() + px(1.), "{row:?}, {hint:?}");
        if width == 640. {
            assert!(
                hint.top() >= button.bottom() && hint.left() == row.left(),
                "{button:?}, {hint:?}"
            );
        } else {
            assert_aligned(button, hint, "long cancellation hint");
        }
    }
}

#[gpui::test]
fn typing_batch_size_updates_the_request_and_preserves_an_in_progress_edit(
    cx: &mut TestAppContext,
) {
    cx.update(install);
    cx.update(|cx| {
        let navigation = cx.global::<AppState>().navigation.clone();
        navigation.update(cx, |navigation, cx| {
            navigation.select(Page::Training);
            cx.notify();
        });
    });
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    cx.simulate_resize(size(px(1280.), px(1000.)));
    cx.run_until_parked();
    let field = bounds(cx, "training-batch-size-field");
    cx.simulate_click(
        point(field.left() + px(12.), field.center().y),
        Modifiers::none(),
    );
    // The upstream keymap initializer uses a process-wide OnceLock, whereas
    // these tests create independent Apps. Dispatch the real SelectAll action
    // directly so another test's App cannot consume its keymap registration.
    cx.dispatch_action(yororen_ui::headless::text_input::SelectAll);
    cx.simulate_input("4");
    cx.update(|_, cx| {
        let Request::Train(params) = cx
            .global::<AppState>()
            .form
            .read(cx)
            .request(Path::new("project"))
        else {
            panic!("training form must produce a training request");
        };
        assert_eq!(params.batch_size, 4);
        assert_eq!(cx.global::<BatchInput>().0.read(cx).value, "4");
    });

    cx.dispatch_action(yororen_ui::headless::text_input::SelectAll);
    cx.simulate_input("3.");
    let input = cx.update(|_, cx| cx.global::<BatchInput>().0.clone());
    cx.update(|window, cx| {
        let ui = cx.global::<AppState>().ui.clone();
        ui.update(cx, |ui, cx| {
            ui.training_details = true;
            cx.notify();
        });
        window.refresh();
    });
    cx.run_until_parked();
    cx.update(|_, cx| {
        assert_eq!(cx.global::<BatchInput>().0.entity_id(), input.entity_id());
        assert_eq!(input.read(cx).value, "3.");
    });
    cx.simulate_input("6");
    cx.update(|window, cx| {
        assert_eq!(input.read(cx).value, "3.6");
        assert_eq!(cx.global::<AppState>().form.read(cx).batch_size(), 4);
        window.blur();
        window.refresh();
    });
    cx.run_until_parked();
    cx.update(|_, cx| assert_eq!(input.read(cx).value, "4"));
}

#[gpui::test]
fn training_bar_counts_steps_within_each_epoch_and_retains_cancelled_progress(
    cx: &mut TestAppContext,
) {
    cx.update(install);
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    for (width, height) in WINDOW_SIZES {
        cx.simulate_resize(size(px(width), px(height)));
        show_task(cx, Page::Training, TaskKind::Train, None);
        cx.update(|window, cx| {
            let form = cx.global::<AppState>().form.clone();
            form.update(cx, |form, cx| {
                // The queued task used 200 epochs and batch size 1. Subsequent
                // edits must not change the running task's denominator.
                form.set_epochs(400.);
                form.set_batch_size(4.);
                cx.notify();
            });
            window.refresh();
        });
        // Last step of epoch 1, first of epoch 2, then a resume/rollback within
        // epoch 2. Literal expected values also catch a modulo-at-boundary bug.
        for (epoch, step, expected) in [
            (0, 7929, 7929.),
            (1, 7930, 1.),
            (1, 8052, 123.),
            (1, 8000, 71.),
        ] {
            cx.update(|window, cx| {
                let tasks = cx.global::<AppState>().tasks.clone();
                let mut event = Event::new(
                    TaskId::parse(TASK_ID).unwrap(),
                    "2026-09-08T00:00:00Z",
                    TaskStage::Training {
                        epoch,
                        step,
                        loss: 0.25,
                    },
                );
                event.progress = Some(Progress {
                    completed: step,
                    total: Some(1_585_800),
                });
                tasks.update(cx, |tasks, cx| {
                    tasks.apply(TaskUpdate::Progress(Box::new(event)));
                    cx.notify();
                });
                window.refresh();
            });
            cx.run_until_parked();
            assert_training_bar(cx, expected, 7929.);
            let epoch_label = bounds(cx, "training-epoch-position");
            let steps_label = bounds(cx, "training-epoch-steps");
            assert_inline_text(epoch_label, 70., "training epoch");
            assert_inline_text(steps_label, 80., "training steps");
            assert_aligned(epoch_label, steps_label, "training progress");
        }
        cx.update(|window, cx| {
            let tasks = cx.global::<AppState>().tasks.clone();
            tasks.update(cx, |tasks, cx| {
                let id = TaskId::parse(TASK_ID).unwrap();
                tasks.apply(TaskUpdate::Progress(Box::new(Event::new(
                    id.clone(),
                    "2026-09-08T00:00:01Z",
                    TaskStage::Cancelled,
                ))));
                tasks.apply(TaskUpdate::Finished(Box::new(SupervisionReport {
                    task_id: id,
                    attempts: 1,
                    outcome: SupervisedOutcome::Cancelled,
                    crash_logs: Vec::new(),
                    journal_errors: Vec::new(),
                })));
                cx.notify();
            });
            window.refresh();
        });
        cx.run_until_parked();
        assert_training_bar(cx, 71., 7929.);
        assert_inline_text(
            bounds(cx, "training-epoch-position"),
            70.,
            "cancelled epoch",
        );
        assert_inline_text(bounds(cx, "training-epoch-steps"), 80., "cancelled steps");
    }
}

#[gpui::test]
fn training_without_an_observed_batch_does_not_claim_a_complete_epoch(cx: &mut TestAppContext) {
    cx.update(install);
    let (_view, cx) = cx.add_window_view(|_, _| Workbench);
    show_task(cx, Page::Training, TaskKind::Train, None);
    cx.update(|_, cx| {
        let rendered = cx
            .global::<RenderedTrainingProgress>()
            .0
            .lock()
            .unwrap()
            .take()
            .unwrap();
        assert!(rendered.indeterminate);
    });
    // A resumed task may finish without executing a batch. Completion alone
    // must not fabricate the last epoch or fill an unknown epoch's bar.
    show_task(
        cx,
        Page::Training,
        TaskKind::Train,
        Some(SupervisedOutcome::Completed { result: None }),
    );
    assert_training_bar(cx, 0., 1.);
}

fn assert_training_bar(cx: &mut VisualTestContext, value: f32, max: f32) {
    cx.update(|_, cx| {
        let rendered = cx
            .global::<RenderedTrainingProgress>()
            .0
            .lock()
            .unwrap()
            .take()
            .expect("training progress must be rendered, including after cancellation");
        assert!(!rendered.indeterminate);
        assert_eq!((rendered.value, rendered.max), (value, max));
    });
}
