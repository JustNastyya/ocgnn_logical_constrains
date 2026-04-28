# Torchlogix Tryout

Exploring using logic gate networks (torchlogix) to generate logical constraints for OCGIN.

## Files

- `tryout.py` - Main script to train on Cora
- `test_torchlogix.py` - Quick installation test

## Setup

```bash
# Install torchlogix (requires Python 3.12, torch 2.6+)
pip install torchlogix
```

## Running

```bash
# Quick test first
python models/logix_gates/test_torchlogix.py

# Then try Cora
python models/logix_gates/tryout.py
```

## Understanding the Output

The model learns logic gates:
- `INPUT` - use feature as-is
- `NOT_INPUT` - negate feature
- `AND`, `OR`, `NAND`, `NOR` - combine features
- `XOR`, `XNOR` - exclusive combinations
- `LESS`, `LEQ`, `GREATER`, `GEQ` - threshold comparisons

## Next Steps

1. Verify it trains on Cora
2. Extract readable formulas
3. Map gate outputs to graph constraints
4. Integrate with OCGIN loss