pub mod http;
pub mod embeddings;
pub mod utils;

use axum::{Json, extract::State};
use std::sync::Arc;
use utoipa::ToSchema;

pub use crate::embeddings::service::EmbeddingService;

#[derive(serde::Deserialize, serde::Serialize, ToSchema)]
pub struct EmbeddingRequest {
    /// The text to generate an embedding for
    pub text: String,
    /// Optional model name, defaults to "text-embedding-3-large"
    pub model: Option<String>,
    /// The type of embedding (e.g., "user", "title", etc.)
    pub embedding_type: String,
}

#[derive(serde::Serialize, ToSchema)]
pub struct StoreResponse {
    /// The generated embedding vector
    pub embedding: Vec<f64>,
    /// Whether the embedding was successfully stored
    pub stored: bool,
}

#[derive(serde::Deserialize, ToSchema)]
pub struct CompareRequest {
    /// The text to compare with stored embeddings
    pub text: String,
    /// Optional model name, defaults to "text-embedding-3-large"
    pub model: Option<String>,
    /// Number of top results to return, defaults to all
    pub top_k: Option<usize>,
    /// Whether to include embeddings in the response
    pub include_embeddings: Option<bool>,
    /// The type of embedding to compare against (e.g., "user", "title", etc.)
    pub embedding_type: Option<String>,
}

#[derive(serde::Serialize, ToSchema)]
pub struct CompareResponse {
    /// List of comparison results, sorted by similarity
    pub results: Vec<ComparisonResult>,
}

#[derive(serde::Serialize, ToSchema)]
pub struct ComparisonResult {
    /// The text that was compared
    pub text: String,
    /// The similarity score
    pub similarity: f64,
    /// The embedding vector, if requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f64>>,
    /// The type of the embedding
    pub embedding_type: String,
}

#[derive(serde::Serialize, ToSchema)]
pub struct ClearResponse {
    /// Whether the data was successfully cleared
    pub success: bool,
}

/// Store a new text embedding
#[utoipa::path(
    post,
    path = "/store",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Embedding successfully stored", body = StoreResponse),
        (status = 500, description = "Failed to generate or store embedding")
    ),
    tag = "embeddings"
)]
pub async fn store_embedding(
    State(embedding_service): State<Arc<EmbeddingService>>,
    Json(payload): Json<EmbeddingRequest>,
) -> Json<StoreResponse> {
    let mut model = payload.model.unwrap_or_else(|| "text-embedding-3-large".to_string());
    // check if model is one of text-embedding-3-large, text-embedding-3-small, text-embedding-3-base
    // else default to text-embedding-3-large
    if model != "text-embedding-3-large" && model != "text-embedding-3-small" && model != "text-embedding-3-base" {
        model = "text-embedding-3-large".to_string();
    }
    // Get embedding
    let embedding_vec = embedding_service.get_embedding(&payload.text, &model).await
        .expect("Failed to get embedding");

    // Save the new embedding
    let store_result = embedding_service.save_embedding(
        &payload.text,
        &embedding_vec,
        &model,
        &payload.embedding_type
    ).await;

    // Check if it was actually stored (not a duplicate)
    let stored = match store_result {
        Ok(_) => true,
        Err(e) => {
            if e.to_string().contains("duplicate") {
                false
            } else {
                panic!("Failed to store embedding: {}", e)
            }
        }
    };

    Json(StoreResponse {
        embedding: embedding_vec,
        stored,
    })
}

/// Compare text with stored embeddings
#[utoipa::path(
    post,
    path = "/compare",
    request_body = CompareRequest,
    responses(
        (status = 200, description = "Comparison results", body = CompareResponse),
        (status = 500, description = "Failed to generate embedding or compare")
    ),
    tag = "embeddings"
)]
pub async fn compare_embedding(
    State(embedding_service): State<Arc<EmbeddingService>>,
    Json(payload): Json<CompareRequest>,
) -> Json<CompareResponse> {
    let mut model = payload.model.unwrap_or_else(|| "text-embedding-3-large".to_string());
    // check if model is one of text-embedding-3-large, text-embedding-3-small, text-embedding-3-base
    // else default to text-embedding-3-large
    if model != "text-embedding-3-large" && model != "text-embedding-3-small" && model != "text-embedding-3-base" {
        model = "text-embedding-3-large".to_string();
    }
    let include_embeddings = payload.include_embeddings.unwrap_or(false);

    // Get embedding for the input text
    let embedding_vec = embedding_service.get_embedding(&payload.text, &model).await
        .expect("Failed to get embedding");

    // Compare with stored embeddings
    let results = embedding_service.compare_embeddings(
        &payload.text,
        &embedding_vec,
        payload.top_k,
        include_embeddings,
        payload.embedding_type
    ).await
        .expect("Failed to compare embeddings");

    Json(CompareResponse {
        results
    })
}

/// Clear all stored embeddings
#[utoipa::path(
    post,
    path = "/clear",
    responses(
        (status = 200, description = "Data successfully cleared", body = ClearResponse),
        (status = 500, description = "Failed to clear data")
    ),
    tag = "embeddings"
)]
pub async fn clear_embeddings(
    State(embedding_service): State<Arc<EmbeddingService>>,
) -> Json<ClearResponse> {
    let result = embedding_service.clear_data();
    Json(ClearResponse {
        success: result.is_ok(),
    })
} 