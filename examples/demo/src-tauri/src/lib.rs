use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // Try to enable features via webview window
            if let Some(window) = app.get_webview_window("main") {
                // Check if we can enable additional webkit settings
                window.eval(
                    r#"
                    // Try setting up the environment for multi-threaded WASM
                    try {
                        // Even if headers can't be set, verify SharedArrayBuffer is available
                        console.log('SharedArrayBuffer:', typeof SharedArrayBuffer !== 'undefined');
                        console.log('hardwareConcurrency:', navigator.hardwareConcurrency);
                    } catch(e) {
                        console.error('Setup error:', e);
                    }
                    "#
                ).ok();
            }
            
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
