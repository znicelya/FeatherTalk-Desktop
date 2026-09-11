//! The FeatherTalk desktop entry point.
//!
//! Bootstrap only, in one fixed order: renderer and theme, the text-input keymap,
//! the app catalog, the global state, then the window. Rendering a themed
//! component before `renderer::install` or reading a global before it is set are
//! the two mistakes this order exists to prevent.

#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use feathertalk_app::args::{self, LaunchOptions};
use feathertalk_app::catalog;
use feathertalk_app::components::compute_control;
use feathertalk_app::state::AppState;
use feathertalk_app::workbench::{Workbench, install_keyboard_navigation};
use feathertalk_app::{theme, ui_assets::WorkbenchAssets};
use gpui::{
    App, AppContext, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px, size,
};
use yororen_ui::i18n::Translate;
use yororen_ui::renderer;

fn main() {
    let launch = launch_options();
    let app = Application::new().with_assets(WorkbenchAssets);
    app.run(move |cx: &mut App| {
        renderer::install_with(cx, theme::palette(false));
        feathertalk_app::components::control_renderers::install(cx);
        // Idempotent, and the asset, training and generate pages all need it.
        yororen_ui::headless::text_input::init(cx);
        install_keyboard_navigation(cx);
        catalog::install(cx, feathertalk_app::ui::AppLocale::default());
        let state = AppState::new(cx, launch);
        cx.set_global(state);
        compute_control::refresh(cx);
        // Design section 11: a quit handler gets 100 ms, which is not enough to
        // write a checkpoint, so the request goes out and the worker finishes on
        // its own. It is a separate process, and the supervision thread is
        // detached, so `Drop for Transport` -- which would kill it mid-write --
        // never runs at process exit. Dropping the subscription would unregister
        // the handler, so it is detached.
        cx.on_app_quit(|cx| {
            let tasks = cx.global::<AppState>().tasks.clone();
            let asked = tasks.read(cx).request_stop();
            if asked > 0 {
                // English on purpose: diagnostic detail, and the window that would
                // show user-facing copy is already gone.
                eprintln!("feathertalk-app: asked {asked} unfinished task(s) to stop");
            }
            std::future::ready(())
        })
        .detach();
        let title = cx.t("shell.title");
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                None,
                size(px(1280.0), px(800.0)),
                cx,
            ))),
            titlebar: Some(TitlebarOptions {
                title: Some(title),
                ..Default::default()
            }),
            window_min_size: Some(size(px(960.0), px(640.0))),
            app_id: Some("dev.feathertalk.app".to_string()),
            ..Default::default()
        };
        if let Err(error) = cx.open_window(options, |_window, cx| cx.new(|_| Workbench)) {
            // English on purpose: this is diagnostic detail, and there is no window
            // left to show user-facing copy in.
            eprintln!("feathertalk-app: the main window could not be opened: {error}");
            cx.quit();
        }
    });
}

/// Read `--project` from the command line.
///
/// A bad flag is reported in English on stderr and the shell starts without a
/// project: the task page then says what cannot be done yet, which beats exiting
/// before anything is on screen. A desktop application that dies on a mistyped
/// switch is the hardest kind of failure to diagnose.
fn launch_options() -> LaunchOptions {
    match args::parse(std::env::args_os().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("feathertalk-app: {error}");
            LaunchOptions::default()
        }
    }
}
