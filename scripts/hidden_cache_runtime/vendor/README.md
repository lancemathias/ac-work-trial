`clef_joint_schema.py` is an unmodified copy of Cloudflare's Apache-2.0-licensed
[Clef implementation](https://huggingface.co/Cloudflare/clef/blob/0b331204bb13fbd2ca93a64df1956f5b55478ce5/joint_schema_model.py).
Revision: `0b331204bb13fbd2ca93a64df1956f5b55478ce5`.
SHA256: `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
The upstream license is retained as `CLEF-LICENSE`.

E1 uses its encoder, collation and fresh `JointSchemaHead`, not its released
trained backbone/head or rounded answer serializer. The encoder includes question
IDs and all schema fields in the prompt, unlike Jev's documented ID-independent
behavior; this is request-shape compatibility, not an implementation clone.
