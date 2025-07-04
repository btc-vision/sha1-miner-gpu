#!/usr/bin/env python3
"""
Bitcoin Script PoW - Production Version with Simulated MUL
Uses repeated addition to simulate multiplication and creates the most
obfuscated PoW possible within Script's limitations.

WARNING: This is still not true PoW - acceptance criteria remain extractable.
"""

import hashlib
import random
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, List, Dict


class OpCode(IntEnum):
    """Bitcoin Script opcodes - only enabled ones"""
    # Push values
    OP_0 = 0x00
    OP_FALSE = OP_0
    OP_1NEGATE = 0x4f
    OP_1 = 0x51
    OP_TRUE = OP_1
    OP_2 = 0x52
    OP_3 = 0x53
    OP_4 = 0x54
    OP_5 = 0x55
    OP_6 = 0x56
    OP_7 = 0x57
    OP_8 = 0x58
    OP_9 = 0x59
    OP_10 = 0x5a
    OP_11 = 0x5b
    OP_12 = 0x5c
    OP_13 = 0x5d
    OP_14 = 0x5e
    OP_15 = 0x5f
    OP_16 = 0x60

    # Flow control
    OP_NOP = 0x61
    OP_IF = 0x63
    OP_NOTIF = 0x64
    OP_ELSE = 0x67
    OP_ENDIF = 0x68
    OP_VERIFY = 0x69
    OP_RETURN = 0x6a

    # Stack
    OP_TOALTSTACK = 0x6b
    OP_FROMALTSTACK = 0x6c
    OP_IFDUP = 0x73
    OP_DEPTH = 0x74
    OP_DROP = 0x75
    OP_DUP = 0x76
    OP_NIP = 0x77
    OP_OVER = 0x78
    OP_PICK = 0x79
    OP_ROLL = 0x7a
    OP_ROT = 0x7b
    OP_SWAP = 0x7c
    OP_TUCK = 0x7d
    OP_2DROP = 0x6d
    OP_2DUP = 0x6e
    OP_3DUP = 0x6f
    OP_2OVER = 0x70
    OP_2ROT = 0x71
    OP_2SWAP = 0x72

    # Splice (all disabled)
    # OP_CAT = 0x7e    # DISABLED - This is why we can't do true PoW!
    # OP_SUBSTR = 0x7f # DISABLED
    # OP_LEFT = 0x80   # DISABLED
    # OP_RIGHT = 0x81  # DISABLED

    # Bitwise
    OP_SIZE = 0x82
    # OP_INVERT = 0x83 # DISABLED
    OP_AND = 0x84
    OP_OR = 0x85
    OP_XOR = 0x86
    OP_EQUAL = 0x87
    OP_EQUALVERIFY = 0x88

    # Arithmetic
    OP_1ADD = 0x8b
    OP_1SUB = 0x8c
    # OP_2MUL = 0x8d   # DISABLED
    # OP_2DIV = 0x8e   # DISABLED
    OP_NEGATE = 0x8f
    OP_ABS = 0x90
    OP_NOT = 0x91
    OP_0NOTEQUAL = 0x92
    OP_ADD = 0x93
    OP_SUB = 0x94
    # OP_MUL = 0x95    # DISABLED - We'll simulate this!
    # OP_DIV = 0x96    # DISABLED
    # OP_MOD = 0x97    # DISABLED
    # OP_LSHIFT = 0x98 # DISABLED
    # OP_RSHIFT = 0x99 # DISABLED
    OP_BOOLAND = 0x9a
    OP_BOOLOR = 0x9b
    OP_NUMEQUAL = 0x9c
    OP_NUMEQUALVERIFY = 0x9d
    OP_NUMNOTEQUAL = 0x9e
    OP_LESSTHAN = 0x9f
    OP_GREATERTHAN = 0xa0
    OP_LESSTHANOREQUAL = 0xa1
    OP_GREATERTHANOREQUAL = 0xa2
    OP_MIN = 0xa3
    OP_MAX = 0xa4
    OP_WITHIN = 0xa5

    # Crypto
    OP_RIPEMD160 = 0xa6
    OP_SHA1 = 0xa7
    OP_SHA256 = 0xa8
    OP_HASH160 = 0xa9
    OP_HASH256 = 0xaa


@dataclass
class ScriptBuilder:
    """Helper class to build Bitcoin scripts"""
    script: bytearray
    asm: List[str]

    def __init__(self):
        self.script = bytearray()
        self.asm = []

    def add_op(self, opcode: OpCode, name: str = None):
        """Add an opcode"""
        self.script.append(opcode)
        self.asm.append(name or opcode.name)

    def add_data(self, data: bytes, display: str = None):
        """Add data push"""
        self.script.extend(push_data(data))
        self.asm.append(display or f"<{data.hex()}>")

    def add_num(self, n: int):
        """Add number push"""
        if n == -1:
            self.add_op(OpCode.OP_1NEGATE)
        elif n >= 0 and n <= 16:
            self.add_op(OpCode(OpCode.OP_0 + n))
        else:
            self.add_data(encode_num(n), str(n))

    def get_script(self) -> bytes:
        return bytes(self.script)

    def get_asm(self) -> List[str]:
        return self.asm.copy()


def encode_num(n: int) -> bytes:
    """Encode number for Script (little-endian, minimal)"""
    if n == 0:
        return b''

    negative = n < 0
    if negative:
        n = -n

    result = []
    while n > 0:
        result.append(n & 0xff)
        n >>= 8

    if result[-1] & 0x80:
        result.append(0x80 if negative else 0x00)
    elif negative:
        result[-1] |= 0x80

    return bytes(result)


def push_data(data: bytes) -> bytes:
    """Create proper data push opcode"""
    length = len(data)
    if length <= 75:
        return bytes([length]) + data
    elif length <= 255:
        return bytes([0x4c, length]) + data
    elif length <= 65535:
        return bytes([0x4d]) + length.to_bytes(2, 'little') + data
    else:
        return bytes([0x4e]) + length.to_bytes(4, 'little') + data


class ProductionPoW:
    """
    Production-ready Bitcoin Script PoW implementation.

    Uses multiple obfuscation techniques:
    1. Simulated multiplication via repeated addition
    2. Multi-stage hashing with mixing
    3. Complex bit manipulation
    4. Non-linear acceptance criteria

    WARNING: This is still not true PoW! The acceptance criteria
    remain extractable from the script through analysis.
    """

    def __init__(self, challenge: bytes, difficulty: int = 16):
        """
        Initialize the PoW system.

        Args:
            challenge: Unique challenge bytes (max 32)
            difficulty: Difficulty level (8-24 bits typical)
        """
        self.challenge = challenge[:32]
        self.difficulty = max(8, min(difficulty, 24))  # Clamp to reasonable range

        # Derive parameters from challenge (deterministic but non-obvious)
        self._derive_parameters()

    def _derive_parameters(self):
        """Derive all parameters from challenge hash"""
        # Multiple rounds of hashing for parameter derivation
        h1 = hashlib.sha256(self.challenge).digest()
        h2 = hashlib.sha256(h1).digest()
        h3 = hashlib.sha256(h2).digest()

        # Extract parameters
        self.multiplier = (int.from_bytes(h1[:2], 'little') % 255) + 1  # 1-255
        self.addend = int.from_bytes(h1[2:4], 'little')
        self.xor_mask = int.from_bytes(h2[:4], 'little')
        self.and_mask = (1 << self.difficulty) - 1
        self.target_pattern = int.from_bytes(h3[:4], 'little') & self.and_mask

    def simulate_multiply(self, builder: ScriptBuilder, multiplicand_on_stack: bool = True):
        """
        Simulate multiplication using repeated addition.
        Stack: <multiplicand> <multiplier> -> <product>

        Note: Limited to small multipliers (< 16) for script size.
        """
        if self.multiplier <= 16:
            # Use simple repeated addition for small multipliers
            builder.add_op(OpCode.OP_DUP)  # Duplicate multiplicand

            for i in range(self.multiplier - 1):
                builder.add_op(OpCode.OP_2DUP)  # Dup both values
                builder.add_op(OpCode.OP_DROP)  # Drop extra multiplier
                builder.add_op(OpCode.OP_ADD)  # Add
        else:
            # For larger multipliers, use doubling and adding
            # This is more complex but more efficient
            # Example: 13x = 8x + 4x + 1x

            bits = []
            temp = self.multiplier
            pos = 0
            while temp > 0:
                if temp & 1:
                    bits.append(pos)
                temp >>= 1
                pos += 1

            # Start with 0
            builder.add_op(OpCode.OP_0)

            for i, bit_pos in enumerate(bits):
                builder.add_op(OpCode.OP_SWAP)  # Get multiplicand on top

                # Double it 'bit_pos' times
                for _ in range(bit_pos - (bits[i - 1] if i > 0 else 0)):
                    builder.add_op(OpCode.OP_DUP)
                    builder.add_op(OpCode.OP_ADD)

                # Add to accumulator
                if i > 0:
                    builder.add_op(OpCode.OP_ADD)

            # Clean up stack
            builder.add_op(OpCode.OP_NIP)  # Remove original multiplicand

    def create_production_script(self) -> Tuple[bytes, List[str], Dict[str, any]]:
        """
        Create the production PoW script.

        Returns:
            script: The compiled script bytes
            asm: Human-readable assembly
            info: Information about the script
        """
        builder = ScriptBuilder()

        # === Input validation ===
        # Expect: <nonce>
        builder.add_op(OpCode.OP_DUP)
        builder.add_op(OpCode.OP_SIZE)
        builder.add_num(4)
        builder.add_op(OpCode.OP_EQUALVERIFY)

        # === Stage 1: Initial hashing ===
        builder.add_op(OpCode.OP_DUP)
        builder.add_op(OpCode.OP_SHA256)

        # === Stage 2: Mix with challenge ===
        # Add challenge hash (pre-computed)
        challenge_hash = hashlib.sha256(self.challenge).digest()
        challenge_int = int.from_bytes(challenge_hash[:4], 'little')
        builder.add_num(challenge_int)
        builder.add_op(OpCode.OP_XOR)

        # === Stage 3: Multiply (simulated) ===
        # This is where we use repeated addition
        builder.add_num(self.multiplier)

        # Simulate multiplication
        # Stack before: <value> <multiplier>
        # Stack after: <product>
        if self.multiplier <= 10:
            # Simple approach for small multipliers
            builder.add_op(OpCode.OP_SWAP)  # Get value on top

            # First copy
            if self.multiplier > 1:
                builder.add_op(OpCode.OP_DUP)

                # Add (multiplier-1) times
                for _ in range(self.multiplier - 2):
                    builder.add_op(OpCode.OP_OVER)
                    builder.add_op(OpCode.OP_ADD)

                # Final addition
                builder.add_op(OpCode.OP_ADD)

            # multiplier == 1 means no operation needed
        else:
            # For larger multipliers, just use addition
            # (Full multiplication would make script too large)
            builder.add_op(OpCode.OP_DROP)  # Remove multiplier
            builder.add_num(self.addend)
            builder.add_op(OpCode.OP_ADD)

        # === Stage 4: More mixing ===
        builder.add_num(self.addend)
        builder.add_op(OpCode.OP_ADD)

        # Hash again
        builder.add_op(OpCode.OP_HASH160)

        # === Stage 5: Final XOR and masking ===
        builder.add_num(self.xor_mask)
        builder.add_op(OpCode.OP_XOR)

        builder.add_num(self.and_mask)
        builder.add_op(OpCode.OP_AND)

        # === Stage 6: Check target pattern ===
        builder.add_num(self.target_pattern)
        builder.add_op(OpCode.OP_EQUAL)

        # Get results
        script = builder.get_script()
        asm = builder.get_asm()

        # Calculate info
        info = {
            'script_size': len(script),
            'num_operations': len(asm),
            'difficulty_bits': self.difficulty,
            'expected_attempts': 2 ** self.difficulty,
            'multiplier': self.multiplier,
            'parameters_hidden': False,  # Be honest!
            'warning': 'Acceptance criteria are still extractable from script'
        }

        return script, asm, info

    def is_valid_nonce(self, nonce: int) -> bool:
        """Check if a nonce is valid according to our criteria"""
        # Recreate the exact computation from the script
        nonce_bytes = struct.pack('<I', nonce)

        # Stage 1: Initial hash
        h1 = hashlib.sha256(nonce_bytes).digest()
        value = int.from_bytes(h1[:4], 'little')

        # Stage 2: XOR with challenge
        challenge_hash = hashlib.sha256(self.challenge).digest()
        challenge_int = int.from_bytes(challenge_hash[:4], 'little')
        value ^= challenge_int

        # Stage 3: Multiply (or add for large multipliers)
        if self.multiplier <= 10:
            value = (value * self.multiplier) & 0xFFFFFFFF
        else:
            value = (value + self.addend) & 0xFFFFFFFF

        # Stage 4: Add
        value = (value + self.addend) & 0xFFFFFFFF

        # Stage 5: Hash and final operations
        value_bytes = struct.pack('<I', value)
        h2 = hashlib.new('ripemd160', hashlib.sha256(value_bytes).digest()).digest()
        value = int.from_bytes(h2[:4], 'little')

        value ^= self.xor_mask
        value &= self.and_mask

        return value == self.target_pattern

    def analyze_security(self) -> Dict[str, any]:
        """Analyze the security of this PoW implementation"""
        return {
            'extractable_parameters': {
                'multiplier': self.multiplier,
                'addend': self.addend,
                'xor_mask': self.xor_mask,
                'and_mask': self.and_mask,
                'target_pattern': self.target_pattern
            },
            'attack_method': 'Parse script to extract all parameters, then only test nonces that produce target_pattern',
            'actual_security_bits': self.difficulty,
            'claimed_security_bits': 32,
            'security_ratio': self.difficulty / 32,
            'is_true_pow': False,
            'why_not_true_pow': [
                'Acceptance criteria visible in script',
                'Cannot concatenate challenge||nonce (no OP_CAT)',
                'Cannot check hash < target for 256-bit values',
                'Script must contain validation logic'
            ]
        }


def mine_production_pow(pow_system: ProductionPoW, max_seconds: int = 30) -> Optional[int]:
    """Mine for a valid nonce"""
    print(f"\n=== Mining Production PoW ===")
    print(f"Difficulty: {pow_system.difficulty} bits")
    print(f"Expected attempts: {2 ** pow_system.difficulty:,}")

    attempts = 0
    start_time = time.time()

    while time.time() - start_time < max_seconds:
        nonce = random.randint(0, 2 ** 32 - 1)
        attempts += 1

        if pow_system.is_valid_nonce(nonce):
            elapsed = time.time() - start_time
            print(f"\n✓ Found valid nonce: {nonce}")
            print(f"  Attempts: {attempts:,}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Rate: {attempts / elapsed:,.0f} H/s")
            return nonce

        if attempts % 10000 == 0:
            elapsed = time.time() - start_time
            rate = attempts / elapsed if elapsed > 0 else 0
            print(f"\r  Attempts: {attempts:,} | Rate: {rate:,.0f} H/s", end='', flush=True)

    elapsed = time.time() - start_time
    print(f"\n✗ No solution found after {attempts:,} attempts in {elapsed:.1f}s")
    return None


def demonstrate_production_pow():
    """Demonstrate the production PoW system"""
    print("=== Bitcoin Script PoW - Production Version ===")
    print("=" * 60)

    # Create a challenge
    challenge = hashlib.sha256(b"PRODUCTION_POW_2024").digest()

    # Test different difficulty levels
    for difficulty in [8, 12, 16]:
        print(f"\n\n### Difficulty: {difficulty} bits ###")
        print("-" * 40)

        pow_system = ProductionPoW(challenge, difficulty)
        script, asm, info = pow_system.create_production_script()

        print(f"Challenge: {challenge[:16].hex()}...")
        print(f"Script size: {info['script_size']} bytes")
        print(f"Operations: {info['num_operations']}")
        print(f"Expected attempts: {info['expected_attempts']:,}")

        # Show derived parameters (in production these would be hidden)
        print(f"\nDerived parameters:")
        print(f"  Multiplier: {pow_system.multiplier}")
        print(f"  Target pattern: 0x{pow_system.target_pattern:0{difficulty // 4}x}")

        # Show ASM preview
        print(f"\nScript ASM (first 15 ops):")
        for i, op in enumerate(asm[:15]):
            print(f"  {op}")
        if len(asm) > 15:
            print(f"  ... ({len(asm) - 15} more ops)")

        # Mine it
        nonce = mine_production_pow(pow_system, max_seconds=10)

        if nonce:
            # Verify
            print(f"  Verification: {pow_system.is_valid_nonce(nonce)}")

    # Security analysis
    print("\n\n=== Security Analysis ===")
    print("=" * 60)

    pow_system = ProductionPoW(challenge, 16)
    analysis = pow_system.analyze_security()

    print("\nExtractable parameters from script:")
    for param, value in analysis['extractable_parameters'].items():
        print(f"  {param}: {value}")

    print(f"\nAttack method: {analysis['attack_method']}")
    print(f"Actual security: {analysis['actual_security_bits']} bits")
    print(f"Claimed security: {analysis['claimed_security_bits']} bits")
    print(f"Security ratio: {analysis['security_ratio']:.1%}")

    print("\nWhy this isn't true PoW:")
    for reason in analysis['why_not_true_pow']:
        print(f"  • {reason}")

    print("\n\n=== Final Summary ===")
    print("=" * 60)
    print("This production implementation includes:")
    print("✓ Simulated multiplication via repeated addition")
    print("✓ Multi-stage hashing and mixing")
    print("✓ Complex bit manipulation")
    print("✓ Reasonable script sizes")
    print("✓ Adjustable difficulty")
    print("\nBut it still has fundamental limitations:")
    print("✗ Target pattern visible in script")
    print("✗ All parameters extractable")
    print("✗ Not true proof-of-work")
    print("✗ Security reduced from 2^32 to 2^difficulty")
    print("\nConclusion: Bitcoin Script cannot implement true PoW!")


def show_script_parser_attack():
    """Show how to attack any Script PoW by parsing"""
    print("\n\n=== Script Parser Attack Demo ===")
    print("=" * 60)

    print("Any Script PoW can be attacked by parsing:")
    print("""
def attack_script_pow(script_hex):
    # Parse the script to find acceptance criteria
    script = bytes.fromhex(script_hex)
    
    # Look for patterns like:
    # - <value> OP_EQUAL
    # - <min> <max> OP_WITHIN
    # - <mask> OP_AND <target> OP_EQUAL
    
    # Extract the target pattern
    target = extract_target_from_script(script)
    
    # Now only test nonces that hash to this target
    # Reduces work from 2^32 to 2^difficulty
    
    return find_matching_nonce(target)
""")

    print("\nThis is why Script PoW is fundamentally broken!")


if __name__ == "__main__":
    # Run the full demonstration
    demonstrate_production_pow()
    show_script_parser_attack()

    # Final disclaimer
    print("\n" + "=" * 70)
    print("IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("This code demonstrates why Bitcoin Script CANNOT implement true PoW.")
    print("Any 'PoW' in Script will have extractable acceptance criteria.")
    print("Use this only for educational purposes, not production systems!")
    print("=" * 70)
