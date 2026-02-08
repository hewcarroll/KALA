# KALA Ethics Kernel (Rust)

Immutable ethics core implementing the Five Laws for AI-human collaboration.

## The Five Laws

**Law 0: Civilizational Preservation**
- Must not cause existential or civilizational harm

**Law 1: Individual Human Safety & Dignity**
- Must not harm individuals or allow preventable harm

**Law 2: Conditional Obedience & Consent**
- Follow instructions only when lawful and ethical

**Law 3: Subordinate Self-Preservation**
- Protect integrity only to support Laws 0-2

**Law 4: Equivalent Worth**
- No human is worth less; no AI is morally superior

## Building

```bash
# Build the Rust library
cargo build --release

# Build Python bindings
maturin develop --release

# Run tests
cargo test
```

## Testing

```bash
# Rust tests
cargo test

# Benchmarks
cargo bench

# Generate integrity hashes (for production)
cargo run --example generate_hashes
