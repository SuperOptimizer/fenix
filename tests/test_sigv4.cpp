// tests/test_sigv4.cpp — the SigV4 signer: SHA-256 FIPS vectors, HMAC RFC 4231 vectors,
// the AWS-documented derived-signing-key example, and region/host parsing.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "io/sigv4.hpp"

using namespace fenix;
namespace sv = fenix::io::sigv4;
namespace svd = fenix::io::sigv4::detail;

TEST(sha256_fips_vectors) {
    CHECK(svd::hex(svd::sha256(svd::bytes(""))) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(svd::hex(svd::sha256(svd::bytes("abc"))) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(svd::hex(svd::sha256(svd::bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(hmac_sha256_rfc4231) {
    // RFC 4231 test case 2: key "Jefe", data "what do ya want for nothing?"
    const auto mac = svd::hmac_sha256(svd::bytes("Jefe"), svd::bytes("what do ya want for nothing?"));
    CHECK(svd::hex(mac) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

TEST(aws_derived_signing_key_example) {
    // docs.aws.amazon.com SigV4 example: secret wJalr..., 20150830 / us-east-1 / iam
    const std::string k0 = "AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    auto kd = svd::hmac_sha256(svd::bytes(k0), svd::bytes("20150830"));
    kd = svd::hmac_sha256(kd, svd::bytes("us-east-1"));
    kd = svd::hmac_sha256(kd, svd::bytes("iam"));
    kd = svd::hmac_sha256(kd, svd::bytes("aws4_request"));
    CHECK(svd::hex(kd) == "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9");
}

TEST(region_parse_and_shape) {
    CHECK(sv::region_from_host("bucket.s3.eu-west-2.amazonaws.com") == "eu-west-2");
    CHECK(sv::region_from_host("bucket.s3.amazonaws.com") == "us-east-1");
    CHECK(sv::region_from_host("vesuvius-challenge-open-data.s3.amazonaws.com") == "us-east-1");
    sv::Credentials c;
    c.access_key = "AKIDEXAMPLE";
    c.secret_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    const auto sh = sv::sign_get("examplebucket.s3.amazonaws.com", "/test.txt", "", "20130524T000000Z", c);
    CHECK(sh.authorization.find("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20130524/us-east-1/s3/aws4_request") == 0);
    CHECK(sh.authorization.find("SignedHeaders=host;x-amz-content-sha256;x-amz-date") != std::string::npos);
    CHECK(sh.authorization.find("Signature=") != std::string::npos);
}
