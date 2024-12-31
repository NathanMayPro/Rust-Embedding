use crate::http::client::make_http_request;
use crate::embeddings::storage::save_embedding_to_jsonl;
use crate::utils::similarity::cosine_similarity;
use crate::ComparisonResult;
use dotenv::dotenv;
use reqwest::Method;
use std::collections::HashMap;
use std::env;
use std::fs;
use serde_json;

pub struct EmbeddingService;

impl EmbeddingService {
    pub fn new() -> Self {
        Self
    }

    fn get_data_path() -> String {
        dotenv().ok();
        // Use a test-specific file if we're running tests
        if std::thread::current().name().map_or(false, |n| n.starts_with("test_")) {
            format!("data/test_{}.jsonl", std::thread::current().name().unwrap())
        } else {
            env::var("DATA_PATH").unwrap_or_else(|_| "data/embeddings.jsonl".to_string())
        }
    }

    pub fn clear_data(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_data_path();
        if fs::metadata(&path).is_ok() {
            fs::remove_file(&path)?;
        }
        // Also remove the parent directory if it's empty
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if fs::metadata(parent).is_ok() {
                if let Ok(entries) = fs::read_dir(parent) {
                    if entries.count() == 0 {
                        fs::remove_dir(parent)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn get_embedding(&self, text: &str, model: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

        let url = "https://api.openai.com/v1/embeddings".to_string();

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));

        let body = serde_json::json!({
            "model": model,
            "input": text
        });

        let response = make_http_request(
            Method::POST,
            &url,
            Some(headers),
            None,
            Some(body.to_string()),
        )
        .await?;
        
        let json_response: serde_json::Value = serde_json::from_str(&response)?;
        let embedding = json_response
            .get("data")
            .and_then(|data| data.get(0))
            .and_then(|first_embedding| first_embedding.get("embedding"))
            .and_then(|embedding| embedding.as_array())
            .ok_or("Failed to parse embedding response")?;

        let embedding_vec: Vec<f64> = embedding
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        Ok(embedding_vec)
    }

    pub async fn compare_embeddings(
        &self,
        text: &str,
        embedding: &Vec<f64>,
        top_k: Option<usize>,
        include_embeddings: bool,
        embedding_type: Option<String>,
    ) -> Result<Vec<ComparisonResult>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(&Self::get_data_path())?;
        let mut similarities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // First, collect all valid entries
        let entries: Vec<_> = content.lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .collect();

        // Then process them
        for entry in entries {
            let stored_text = entry["text"].as_str().unwrap_or_default();
            let stored_type = entry["embedding_type"].as_str().unwrap_or_default();
            
            // Skip if we've already processed this text+type combination
            let key = format!("{}:{}", stored_text, stored_type);
            if !seen.insert(key) {
                continue;
            }

            // Skip self-comparison
            if stored_text == text && embedding_type.as_ref().map(|t| t.as_str()) == Some(stored_type) {
                continue;
            }

            // Apply type filter if specified
            if let Some(ref target_type) = embedding_type {
                if stored_type != target_type {
                    continue;
                }
            }

            // Get and compare embeddings
            if let Ok(stored_embedding) = serde_json::from_value::<Vec<f64>>(entry["embedding"].clone()) {
                let similarity = cosine_similarity(embedding, &stored_embedding);
                
                let result = ComparisonResult {
                    text: stored_text.to_string(),
                    similarity,
                    embedding: if include_embeddings {
                        Some(stored_embedding)
                    } else {
                        None
                    },
                    embedding_type: stored_type.to_string(),
                };
                similarities.push(result);
            }
        }

        // Sort by similarity
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Apply top_k filter
        if let Some(k) = top_k {
            similarities.truncate(k);
        }

        if similarities.is_empty() && embedding_type.is_none() {
            return Err("No similar embeddings found".into());
        }

        Ok(similarities)
    }

    pub async fn save_embedding(
        &self,
        text: &str,
        embedding: &Vec<f64>,
        model_name: &str,
        embedding_type: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        save_embedding_to_jsonl(text, embedding, &Self::get_data_path(), model_name, embedding_type).await
    }
} 