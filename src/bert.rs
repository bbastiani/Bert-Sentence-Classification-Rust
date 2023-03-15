use tch;
use tch::TchError;
use tch::{Device, Tensor};

use thiserror::Error;

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{EncodeInput, decoders, Model, TokenizerImpl, PaddingParams, TruncationParams};


type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;

#[derive(Error, Debug)]
pub enum BertErrors {
    #[error("Error in tokenization")]
    TokenizerError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Error in torch model")]
    TorchError(#[from] TchError),
}

pub struct BertModel {
    model: tch::CModule,
    tokenizer: BertTokenizer,
    device: Device,
}

impl BertModel {
    pub fn new(model_filename: &str, tokenizer_filename: &str) -> Result<BertModel, BertErrors> {
        let model = load_model(model_filename)?;
        let tokenizer = load_tokenizer(tokenizer_filename)?;
        let device = Device::cuda_if_available();
        Ok(BertModel { model, tokenizer, device })
    }

    pub fn predict(&self, sentence: &str) -> Result<Vec<i64>, BertErrors> {
        let sent: EncodeInput = (*sentence).into();
        let input = self.tokenizer.encode(sent, true)?;
        let ids = input.get_ids().iter().map(|id| *id as i64).collect::<Vec<i64>>(); // from &[u32] to Vec<i64>
        let masks = input.get_attention_mask().iter().map(|mask| *mask as i64).collect::<Vec<i64>>(); // from &[u32] to Vec<i64>
        let input_id = Tensor::of_slice(ids.as_slice()).to_device(self.device);
        let att_mask = Tensor::of_slice(masks.as_slice()).to_device(self.device);
        let output = self.model.forward_ts(&[input_id.unsqueeze(0), att_mask.unsqueeze(0)])?;
        Ok(output.argmax(-1, false).squeeze().into())
    }
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

fn load_model(model_filename: &str) -> Result<tch::CModule, TchError> {
    let model = tch::CModule::load(model_filename)?;
    Ok(model)
}

fn load_tokenizer(tokenizer_filename: &str) -> Result<BertTokenizer, BertErrors> {
    let wp = WordPiece::from_file(&tokenizer_filename).build()?;
    let tokenizer = create_bert_tokenizer(wp);
    Ok(tokenizer)
}

