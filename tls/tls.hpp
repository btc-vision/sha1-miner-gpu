#pragma once
#include <openssl/ssl.h>

namespace ssl = boost::asio::ssl;

class ChromeTLSConfig {
public:
    static void configureContext(ssl::context& ctx) {
        SSL_CTX* native_ctx = ctx.native_handle();

        // Enable TLS 1.2 and 1.3
        SSL_CTX_set_min_proto_version(native_ctx, TLS1_2_VERSION);
        SSL_CTX_set_max_proto_version(native_ctx, TLS1_3_VERSION);

        // Chrome's cipher list for TLS 1.2
        const char* cipher_list =
            "ECDHE-ECDSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES128-GCM-SHA256:"
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-AES128-SHA:"
            "ECDHE-RSA-AES256-SHA:"
            "AES128-GCM-SHA256:"
            "AES256-GCM-SHA384:"
            "AES128-SHA:"
            "AES256-SHA";

        SSL_CTX_set_cipher_list(native_ctx, cipher_list);

        // TLS 1.3 cipher suites (Chrome's preference order)
        SSL_CTX_set_ciphersuites(native_ctx,
            "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256");

        // Set elliptic curves (Chrome's preference)
        SSL_CTX_set1_curves_list(native_ctx, "X25519:P-256:P-384");

        // Chrome's SSL options
        SSL_CTX_set_options(native_ctx,
            SSL_OP_NO_SSLv2 |
            SSL_OP_NO_SSLv3 |
            SSL_OP_NO_COMPRESSION |
            SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION |
            SSL_OP_SINGLE_ECDH_USE |
            SSL_OP_NO_TICKET |  // Chrome manages tickets separately
            SSL_OP_CIPHER_SERVER_PREFERENCE);  // Let server choose cipher

        // Session configuration
        SSL_CTX_set_session_cache_mode(native_ctx,
            SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_NO_INTERNAL_STORE);

        // Set signature algorithms (Chrome's preferences)
        const char* sigalgs =
            "ecdsa_secp256r1_sha256:"
            "rsa_pss_rsae_sha256:"
            "rsa_pkcs1_sha256:"
            "ecdsa_secp384r1_sha384:"
            "rsa_pss_rsae_sha384:"
            "rsa_pkcs1_sha384:"
            "rsa_pss_rsae_sha512:"
            "rsa_pkcs1_sha512:"
            "rsa_pkcs1_sha1";

        SSL_CTX_set1_sigalgs_list(native_ctx, sigalgs);

        // ALPN callback for HTTP/2 and HTTP/1.1
        SSL_CTX_set_alpn_select_cb(native_ctx, alpnSelectProto, nullptr);
    }

    static void configureSSLStream(SSL* ssl, const std::string& hostname) {
        // Enable SNI (Server Name Indication) - crucial for Cloudflare
        SSL_set_tlsext_host_name(ssl, hostname.c_str());

        // Set ALPN protocols (HTTP/2 and HTTP/1.1)
        const unsigned char alpn_protos[] = {
            //2, 'h', '2',                               // HTTP/2
            8, 'h', 't', 't', 'p', '/', '1', '.', '1' // HTTP/1.1
        };
        SSL_set_alpn_protos(ssl, alpn_protos, sizeof(alpn_protos));

        // Enable OCSP stapling
        SSL_set_tlsext_status_type(ssl, TLSEXT_STATUSTYPE_ocsp);

        // Additional Chrome-like options
        SSL_set_options(ssl, SSL_OP_TLSEXT_PADDING);

        // Set supported groups at SSL level too
        SSL_set1_curves_list(ssl, "X25519:P-256:P-384");
    }

private:
    static int alpnSelectProto(SSL* ssl, const unsigned char** out, unsigned char* outlen,
                              const unsigned char* in, unsigned int inlen, void* arg) {
        // Chrome's ALPN negotiation order
        const unsigned char h2[] = { 2, 'h', '2' };
        const unsigned char http11[] = { 8, 'h', 't', 't', 'p', '/', '1', '.', '1' };

        // Try HTTP/2 first
        if (SSL_select_next_proto((unsigned char**)out, outlen, h2, sizeof(h2), in, inlen) == OPENSSL_NPN_NEGOTIATED) {
            return SSL_TLSEXT_ERR_OK;
        }

        // Fall back to HTTP/1.1
        if (SSL_select_next_proto((unsigned char**)out, outlen, http11, sizeof(http11), in, inlen) == OPENSSL_NPN_NEGOTIATED) {
            return SSL_TLSEXT_ERR_OK;
        }

        return SSL_TLSEXT_ERR_NOACK;
    }
};