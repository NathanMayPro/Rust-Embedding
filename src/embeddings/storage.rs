use std::fs::OpenOptions;
use std::io::Write;
use serde_json;


pub async fn save_embedding_to_jsonl(
    text: &str, 
    embedding: &Vec<f64>,
    output_file: &str,
    model_name: &str,
    embedding_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_exists = std::path::Path::new(output_file).exists();
    let mut existing_entries = Vec::new();
    
    if file_exists {
        let content = std::fs::read_to_string(output_file)?;
        for line in content.lines() {
            let entry: serde_json::Value = serde_json::from_str(line)?;
            existing_entries.push(entry);
        }
    }

    let is_duplicate = existing_entries.iter().any(|entry| {
        entry["text"].as_str() == Some(text) && 
        entry["embedding_type"].as_str() == Some(embedding_type)
    });

    if is_duplicate {
        return Err(format!("duplicate text entry for type {}", embedding_type).into());
    }

    let record = serde_json::json!({
        "text": text,
        "embedding": embedding,
        "model": model_name,
        "embedding_type": embedding_type
    });

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_file)?;

    writeln!(file, "{}", record.to_string())?;
    Ok(())
} 