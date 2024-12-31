use reqwest::{Client, Method, Request, Url};
use std::collections::HashMap;

pub async fn make_http_request(
    method: Method,
    url: &str,
    headers: Option<HashMap<String, String>>,
    query_params: Option<HashMap<String, String>>,
    body: Option<String>,
) -> Result<String, reqwest::Error> {
    let client = Client::new();

    let url = Url::parse(url).unwrap();
    let url = if let Some(query_params) = query_params {
        let mut url = url.clone();
        url.query_pairs_mut()
            .extend_pairs(query_params.iter());
        url
    } else {
        url
    };

    let mut request = Request::new(method, url);

    if let Some(headers) = headers {
        for (key, value) in headers {
            request.headers_mut().insert(
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                reqwest::header::HeaderValue::from_str(&value).unwrap(),
            );
        }
    }

    if let Some(body) = body {
        *request.body_mut() = Some(body.into());
    }

    let response = client.execute(request).await?;
    let response_body = response.text().await?;

    Ok(response_body)
} 