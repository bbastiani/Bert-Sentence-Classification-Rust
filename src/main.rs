
mod logging;
mod utils;
mod bert;

use log::{error, info};
use clap::Parser;
use kdam::tqdm;

use bert::BertModel;
use logging::setup_log;
use utils::{save_csv, read_csv};

use polars::prelude::*;

#[derive(Parser)]
struct Args {
    #[arg(short = 'm', long = "model")]
    model_file: String,
    #[arg(short = 't', long = "tokenizer")]
    tokenizer_file: String,
    #[arg(short = 'i', long = "input")]
    sentence_file: String,
    #[arg(short = 'o', long = "output")]
    output_file: String,
}


/// main function
/// Args:
///    model_file: path to model file
///    tokenizer_file: path to vocab file
///    sentence_file: path to txt file containing sentences to be predicted
///    output_file: path to output file to save results
fn main() -> Result<(), String> {
    let args = Args::parse();
    // log start program
    info!("Start program");
    // setup log file
    if let Err(e) = setup_log() {
        return Err(e.to_string());
    }

    let bert = match BertModel::new(&args.model_file, &args.tokenizer_file) {
        Ok(bert) => bert,
        Err(e) => {
            error!("Error in loading model");
            return Err(e.to_string());
        }
    };

    let mut dataframe = match read_csv(&args.sentence_file) {
        Ok(dataframe) => dataframe,
        Err(e) => {
            error!("Error in reading file");
            return Err(e.to_string());
        }
    };
       
    let mut outputs = vec![];
    // iterate through polars dataframe rows
    let sentences = match dataframe.column("sentence") {
        Ok(sentences) => sentences.clone(),
        Err(e) => {
            error!("Error in reading column");
            return Err(e.to_string());
        }
    };

    for sentence in tqdm!(sentences.utf8().unwrap().into_iter().flatten()) {
        let output = match bert.predict(sentence) {
            Ok(output) => output,
            Err(e) => {
                error!("Error in predicting sentence");
                return Err(e.to_string());
            }
        };
        outputs.push(output);
    }
    // add output to dataframe
    dataframe.with_column(
        Series::new("labels", outputs.into_iter().flatten().collect::<Vec<i64>>()),
    );
    // save results
    match save_csv(&args.output_file, dataframe) {
        Ok(_) => (),
        Err(e) => {
            error!("Error in saving results");
            return Err(e.to_string());
        }
    }
    Ok(())
}
