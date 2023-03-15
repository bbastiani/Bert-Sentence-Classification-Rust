use log::SetLoggerError;
use log4rs::{
    append::file::FileAppender,
    config::{runtime::ConfigErrors, Appender, Config, Root},
    encode::pattern::PatternEncoder,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoggingError {
    #[error("Config file not found")]
    FileNotFound(#[from] std::io::Error),
    #[error("Setup Logger Error")]
    LogError(#[from] SetLoggerError),
    #[error("Log Config Error")]
    ConfigLogError(#[from] ConfigErrors),
}

pub fn setup_log() -> Result<(), LoggingError> {
    let level = log::LevelFilter::Info;
    let file_path = "log/log.log";

    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S)} - {l} - {M} - {m}{n}",
        )))
        .build(file_path)?;

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder().appender("logfile").build(level))?;

    let _handle = log4rs::init_config(config)?;

    Ok(())
}
