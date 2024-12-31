# Rust-Embedding Embeddings Service

A high-performance REST API service built in Rust for managing and comparing text embeddings. This service provides endpoints for storing, comparing, and managing text embeddings with support for different embedding types and models.

## Features

- ✨ Store text embeddings with type categorization
- 🔍 Compare text against stored embeddings
- 🎯 Filter comparisons by embedding type
- 📊 Configurable top-k results
- 🚀 Built with Axum for high performance
- 📖 Interactive Swagger UI documentation
- 🔄 Duplicate prevention system
- 🎨 Support for different embedding models

## Prerequisites

- Rust (latest stable version)
- OpenAI API key (for embeddings generation)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Rust-Embedding.git
cd Rust-Embedding
```

2. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
PORT=3000  # Optional, defaults to 3000
```

3. Build and run the project:
```bash
cargo build
cargo run
```

The server will start at `http://0.0.0.0:3000` with Swagger UI documentation available at `http://0.0.0.0:3000/swagger-ui/`.

## API Endpoints

### Store Embedding
```http
POST /store
Content-Type: application/json

{
    "text": "Your text here",
    "model": "text-embedding-3-large",  // Optional
    "embedding_type": "your_type"
}
```

### Compare Embeddings
```http
POST /compare
Content-Type: application/json

{
    "text": "Text to compare",
    "model": "text-embedding-3-large",  // Optional
    "top_k": 5,                        // Optional
    "include_embeddings": true,        // Optional
    "embedding_type": "your_type"      // Optional
}
```

### Clear Embeddings
```http
POST /clear
```

## Testing

Run the test suite with:
```bash
cargo test
```

The project includes comprehensive integration tests covering:
- Embedding storage
- Comparison functionality
- Type-based filtering
- Duplicate prevention
- Default model handling

## Technical Details

- Built with Axum web framework
- Uses OpenAI's text embedding models
- Supports concurrent requests with Arc and async/await
- Implements proper error handling and validation
- Includes Swagger documentation via utoipa

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 