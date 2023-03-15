use thiserror::Error;
use polars::prelude::*;

/// read csv file into a dataframe
pub fn read_csv(filename: &str) -> Result<polars::prelude::DataFrame, PolarsError> {
    Ok(
        CsvReader::from_path(filename)?
            .has_header(true)
            .with_delimiter(b'\t')
            .with_encoding(CsvEncoding::Utf8)
            .finish()?
    )
}

/// save dataframe to csv file
pub fn save_csv(filename: &str, mut dataframe: DataFrame) -> Result<(), PolarsError> {
    let mut file = std::fs::File::create(filename)?;
    CsvWriter::new(&mut file)
    .has_header(true)
    .with_delimiter(b'\t')
    .finish(&mut dataframe)?;
    Ok(())
}