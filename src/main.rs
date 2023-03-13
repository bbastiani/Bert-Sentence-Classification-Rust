use tch;
use tch::TchError;
use tch::{Device, Tensor};

use clap::Parser;
use thiserror::Error;

use csv::Writer;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use kdam::tqdm;

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{decoders, EncodeInput, Model, TokenizerImpl, PaddingParams, TruncationParams};


type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;


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


#[derive(Error, Debug)]
enum Errors {
    #[error("Error open file")]
    FileError(#[from] std::io::Error),
    #[error("Error in tokenization")]
    TokenizerError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Error in torch model")]
    TorchError(#[from] TchError),
    #[error("Error in csv")]
    CsvError(#[from] csv::Error),
}

/// Resembling the BertTokenizer implementation from the Python bindings.
fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
    let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
    let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
    let mut tokenizer = TokenizerImpl::new(wp);
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    tokenizer.with_normalizer(BertNormalizer::default());
    tokenizer.with_decoder(decoders::wordpiece::WordPiece::default());
    tokenizer.with_post_processor(BertProcessing::new(
        ("[SEP]".to_string(), sep_id),
        ("[CLS]".to_string(), cls_id),
    ));
    tokenizer.with_padding(Some(PaddingParams::default()));
    tokenizer.with_truncation(Some(TruncationParams::default()));
    tokenizer
}

/// read file 
fn read_file(filename: &str) -> Result<Vec<EncodeInput>, Errors>{
    let file = File::open(Path::new(filename))?;
    Ok(
        BufReader::new(file)
        .lines()
        .map(|line| line.unwrap().into())
        .collect()
    )
}

/// save results to csv file
fn save_results(filename: &str, results: Vec<Vec<i64>>) -> Result<(), Errors> {
    let mut wtr = Writer::from_path(filename)?;
    for result in results {
        wtr.write_record(result.iter().map(|r| r.to_string()).collect::<Vec<String>>())?;
    }
    wtr.flush()?;
    Ok(())
}

/// main function
/// Args:
///    model_file: path to model file
///    tokenizer_file: path to vocab file
///    sentence_file: path to txt file containing sentences to be predicted
///    output_file: path to output file to save results
fn main() -> Result<(), Errors> {
    let args = Args::parse();
    // create tokenizer
    let wp = WordPiece::from_file(&args.tokenizer_file)
        .build()?;
    let tokenizer = create_bert_tokenizer(wp);
    // load file
    let sentences = read_file(&args.sentence_file)?;
    // load model
    let model = tch::CModule::load(args.model_file)?;
    // predict 
    let device = Device::Cpu;
    let mut outputs = vec![];
    for sentence in tqdm!(sentences.iter()) {
        let encoded_sent = tokenizer.encode((*sentence).clone(), true)?;
        let input_id_array = encoded_sent.get_ids().iter().map(|id| *id as i64).collect::<Vec<i64>>();
        let att_mask_array = encoded_sent.get_attention_mask().iter().map(|mask| *mask as i64).collect::<Vec<i64>>();
        let input_id = Tensor::of_slice(input_id_array.as_slice()).to_device(device);
        let att_mask = Tensor::of_slice(att_mask_array.as_slice()).to_device(device);
        let output = model.forward_ts(&[input_id.unsqueeze(0), att_mask.unsqueeze(0)])?;
        outputs.push(output.argmax(-1, false).squeeze());
    }
    // save results
    save_results(&args.output_file, 
                    outputs.iter()
                        .map(|output| output.into())
                        .collect::<Vec<_>>()
                )?;

    Ok(())
}
