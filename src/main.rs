use axum::{routing::post, Router};
use dotenv::dotenv;
use std::sync::Arc;
use tokio::net::TcpListener;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use rust_embedding::{
    embeddings::service::EmbeddingService,
    store_embedding,
    compare_embedding,
    clear_embeddings,
    EmbeddingRequest,
    CompareRequest,
    StoreResponse,
    CompareResponse,
    ClearResponse,
};

#[derive(OpenApi)]
#[openapi(
    paths(
        rust_embedding::store_embedding,
        rust_embedding::compare_embedding,
        rust_embedding::clear_embeddings
    ),
    components(
        schemas(
            EmbeddingRequest,
            CompareRequest,
            StoreResponse,
            CompareResponse,
            ClearResponse
        )
    ),
    tags(
        (name = "embeddings", description = "Embedding management endpoints")
    ),
    info(
        title = "Embeddings API",
        version = "1.0",
        description = "API for managing and comparing text embeddings"
    )
)]
struct ApiDoc;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let embedding_service = Arc::new(EmbeddingService::new());
    
    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/store", post(store_embedding))
        .route("/compare", post(compare_embedding))
        .route("/clear", post(clear_embeddings))
        .with_state(embedding_service);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("Server running on http://0.0.0.0:{}", port);
    println!("API documentation available at http://0.0.0.0:{}/swagger-ui/", port);
    axum::serve(listener, app).await.unwrap();
} 
