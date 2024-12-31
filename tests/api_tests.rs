use rust_embedding::embeddings;
use rust_embedding::{store_embedding, compare_embedding, clear_embeddings};
use axum::{Router, routing::post};
use std::sync::Arc;
use tokio::net::TcpListener;
use reqwest;
use serde_json::{json, Value};

async fn spawn_app() -> String {
    let embedding_service = Arc::new(embeddings::service::EmbeddingService::new());
    
    let app = Router::new()
        .route("/store", post(store_embedding))
        .route("/compare", post(compare_embedding))
        .route("/clear", post(clear_embeddings))
        .with_state(embedding_service);

    // Bind to a random available port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    base_url
}

#[tokio::test]
async fn test_store_embedding() {
    let base_url = spawn_app().await;
    
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/store", base_url))
        .json(&json!({
            "text": "Hello world",
            "model": "text-embedding-3-large",
            "embedding_type": "test"
        }))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    
    let body: Value = response.json().await.unwrap();
    assert!(body.get("embedding").is_some());
    assert!(body.get("stored").is_some());
    assert!(body["stored"].as_bool().unwrap());
}

#[tokio::test]
async fn test_compare_embeddings() {
    let base_url = spawn_app().await;
    let client = reqwest::Client::new();

    // Clear any existing data
    let clear_response = client
        .post(format!("{}/clear", base_url))
        .send()
        .await
        .unwrap();
    assert!(clear_response.status().is_success());
    
    // Store embeddings with different types
    let store_responses = vec![
        ("This is a user bio", "user"),
        ("Another user bio", "user"),
        ("This is a product title", "title"),
        ("Another product title", "title"),
    ];

    // Keep track of what we store
    let mut stored_texts = Vec::new();
    let mut stored_types = Vec::new();

    for (text, embedding_type) in store_responses {
        let store_response = client
            .post(format!("{}/store", base_url))
            .json(&json!({
                "text": text,
                "model": "text-embedding-3-large",
                "embedding_type": embedding_type
            }))
            .send()
            .await
            .unwrap();
        assert!(store_response.status().is_success());
        let body: Value = store_response.json().await.unwrap();
        assert!(body["stored"].as_bool().unwrap());
        stored_texts.push(text);
        stored_types.push(embedding_type);
    }

    // Test comparing within the same type (user)
    let compare_user = client
        .post(format!("{}/compare", base_url))
        .json(&json!({
            "text": "A new user biography",
            "model": "text-embedding-3-large",
            "top_k": 5,
            "include_embeddings": true,
            "embedding_type": "user"
        }))
        .send()
        .await
        .unwrap();

    let body: Value = compare_user.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty(), "Should have user type results");
    // Verify all results are of type "user"
    for result in results {
        assert_eq!(result["embedding_type"].as_str().unwrap(), "user");
        assert!(stored_texts.contains(&result["text"].as_str().unwrap()));
    }

    // Test comparing within the same type (title)
    let compare_title = client
        .post(format!("{}/compare", base_url))
        .json(&json!({
            "text": "A new product title",
            "model": "text-embedding-3-large",
            "top_k": 5,
            "include_embeddings": true,
            "embedding_type": "title"
        }))
        .send()
        .await
        .unwrap();

    let body: Value = compare_title.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty(), "Should have title type results");
    // Verify all results are of type "title"
    for result in results {
        assert_eq!(result["embedding_type"].as_str().unwrap(), "title");
        assert!(stored_texts.contains(&result["text"].as_str().unwrap()));
    }

    // Test comparing across all types (no type filter)
    let compare_all = client
        .post(format!("{}/compare", base_url))
        .json(&json!({
            "text": "Something to compare with everything",
            "model": "text-embedding-3-large",
            "include_embeddings": true
        }))
        .send()
        .await
        .unwrap();

    let body: Value = compare_all.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty(), "Should have results when comparing across all types");
    
    // Debug print all results
    println!("Total results: {}", results.len());
    for (i, result) in results.iter().enumerate() {
        let text = result["text"].as_str().unwrap();
        let type_ = result["embedding_type"].as_str().unwrap();
        println!("Result {}: type={}, text={}", i, type_, text);
        // Verify this result was one we stored
        assert!(stored_texts.contains(&text), "Found unexpected text: {}", text);
        assert!(stored_types.contains(&type_), "Found unexpected type: {}", type_);
    }
    
    // Count the occurrences of each type
    let user_count = results.iter()
        .filter(|r| r["embedding_type"].as_str().unwrap() == "user")
        .count();
    let title_count = results.iter()
        .filter(|r| r["embedding_type"].as_str().unwrap() == "title")
        .count();
    
    println!("User count: {}, Title count: {}", user_count, title_count);
    
    assert!(user_count > 0, "Should have at least one user type result");
    assert!(title_count > 0, "Should have at least one title type result");
    assert_eq!(user_count + title_count, results.len(), "All results should be either user or title type");
}

#[tokio::test]
async fn test_default_model() {
    let base_url = spawn_app().await;
    
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/store", base_url))
        .json(&json!({
            "text": "Testing default model",
            "embedding_type": "test"
        }))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    
    let body: Value = response.json().await.unwrap();
    assert!(body.get("embedding").is_some());
}

#[tokio::test]
async fn test_duplicate_prevention() {
    let base_url = spawn_app().await;
    let client = reqwest::Client::new();

    // Clear any existing data
    let clear_response = client
        .post(format!("{}/clear", base_url))
        .send()
        .await
        .unwrap();
    assert!(clear_response.status().is_success());

    // Store first embedding
    let first_response = client
        .post(format!("{}/store", base_url))
        .json(&json!({
            "text": "Duplicate text",
            "embedding_type": "test_type"
        }))
        .send()
        .await
        .unwrap();
    assert!(first_response.status().is_success());
    assert!(first_response.json::<Value>().await.unwrap()["stored"].as_bool().unwrap());

    // Try to store the same text with same type
    let duplicate_response = client
        .post(format!("{}/store", base_url))
        .json(&json!({
            "text": "Duplicate text",
            "embedding_type": "test_type"
        }))
        .send()
        .await
        .unwrap();
    assert!(duplicate_response.status().is_success());
    assert!(!duplicate_response.json::<Value>().await.unwrap()["stored"].as_bool().unwrap());

    // Store same text with different type (should succeed)
    let different_type_response = client
        .post(format!("{}/store", base_url))
        .json(&json!({
            "text": "Duplicate text",
            "embedding_type": "different_type"
        }))
        .send()
        .await
        .unwrap();
    assert!(different_type_response.status().is_success());
    assert!(different_type_response.json::<Value>().await.unwrap()["stored"].as_bool().unwrap());
} 