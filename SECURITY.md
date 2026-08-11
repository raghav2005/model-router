# Security policy

Please report suspected vulnerabilities using a private GitHub security advisory after the repository is published. Do not open a public issue containing credentials, customer prompts, provider responses, or exploit details.

The default runtime is shadow mode. Before accepting traffic:

- set API and metrics bearer tokens through a secret manager;
- set a high-entropy audit HMAC key;
- restrict network access so only the router can reach Switchyard;
- keep provider credentials in Switchyard rather than this repository;
- terminate TLS at an approved ingress or service mesh;
- apply tenant authentication, quotas, and rate limits at the ingress layer; and
- complete every gate in `config/release_policy.json`.

The project does not store prompt or response content in its audit log. The optional live evaluation harness can store responses only when explicitly requested; those files must be handled as sensitive evaluation data.
