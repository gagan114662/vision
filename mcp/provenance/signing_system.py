"""
Genuine Provenance Signing and Persistence System.

This implements cryptographic signing of all trading decisions, model outputs,
and system actions with tamper-proof persistence and audit trails.
Addresses the vision requirement for "provenance signing/persistence".
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sqlite3
from contextlib import contextmanager

# Cryptographic signing imports
try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Tamper-proof provenance record with cryptographic signature."""
    record_id: str
    timestamp: datetime
    actor_id: str  # Agent, user, or system component
    action_type: str
    action_description: str
    input_data_hash: str
    output_data_hash: str
    model_version: Optional[str] = None
    configuration_hash: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Related record IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    signature_algorithm: str = "RSA-PSS"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceRecord':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ProvenanceChain:
    """Chain of related provenance records showing decision lineage."""
    chain_id: str
    root_record_id: str
    records: List[ProvenanceRecord] = field(default_factory=list)
    chain_hash: Optional[str] = None

    def add_record(self, record: ProvenanceRecord) -> None:
        """Add record to chain and update chain hash."""
        self.records.append(record)
        self._update_chain_hash()

    def _update_chain_hash(self) -> None:
        """Update the chain hash with all record hashes."""
        if not self.records:
            self.chain_hash = None
            return

        record_hashes = [r.record_id for r in self.records]
        chain_data = f"{self.chain_id}:{':'.join(record_hashes)}"
        self.chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()


class CryptographicSigner:
    """Handles cryptographic signing of provenance records."""

    def __init__(self, key_dir: Optional[Path] = None):
        self.key_dir = key_dir or Path("keys/provenance")
        self.key_dir.mkdir(parents=True, exist_ok=True)

        self.private_key_path = self.key_dir / "private_key.pem"
        self.public_key_path = self.key_dir / "public_key.pem"

        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available, using mock signatures")
            self._mock_mode = True
        else:
            self._mock_mode = False
            self._ensure_keys_exist()

    def _ensure_keys_exist(self) -> None:
        """Generate RSA key pair if it doesn't exist."""
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            logger.info("Generating new RSA key pair for provenance signing")

            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )

            # Get public key
            public_key = private_key.public_key()

            # Save private key
            with open(self.private_key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            # Save public key
            with open(self.public_key_path, "wb") as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))

            logger.info(f"RSA key pair saved to {self.key_dir}")

    def sign_record(self, record: ProvenanceRecord) -> str:
        """Sign a provenance record and return signature."""
        if self._mock_mode:
            return self._mock_sign(record)

        try:
            # Load private key
            with open(self.private_key_path, "rb") as f:
                private_key = load_pem_private_key(f.read(), password=None)

            # Create signable data (exclude signature field)
            signable_data = {
                k: v for k, v in record.to_dict().items()
                if k not in ['signature']
            }
            message = json.dumps(signable_data, sort_keys=True).encode()

            # Sign with RSA-PSS
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            # Return base64 encoded signature
            import base64
            return base64.b64encode(signature).decode()

        except Exception as e:
            logger.error(f"Failed to sign record: {e}")
            return self._mock_sign(record)

    def verify_signature(self, record: ProvenanceRecord) -> bool:
        """Verify the signature of a provenance record."""
        if self._mock_mode:
            return self._mock_verify(record)

        try:
            # Load public key
            with open(self.public_key_path, "rb") as f:
                public_key = load_pem_public_key(f.read())

            # Recreate signable data
            signable_data = {
                k: v for k, v in record.to_dict().items()
                if k not in ['signature']
            }
            message = json.dumps(signable_data, sort_keys=True).encode()

            # Decode signature
            import base64
            signature_bytes = base64.b64decode(record.signature.encode())

            # Verify signature
            public_key.verify(
                signature_bytes,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def _mock_sign(self, record: ProvenanceRecord) -> str:
        """Mock signature for testing when crypto is not available."""
        data = record.to_dict()
        data_str = json.dumps(data, sort_keys=True)
        return f"MOCK_SIGNATURE_{hashlib.md5(data_str.encode()).hexdigest()}"

    def _mock_verify(self, record: ProvenanceRecord) -> bool:
        """Mock signature verification."""
        expected = self._mock_sign(record)
        return record.signature == expected


class ProvenancePersistence:
    """Tamper-proof persistence for provenance records."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("provenance.db")
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with proper schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_description TEXT NOT NULL,
                    input_data_hash TEXT NOT NULL,
                    output_data_hash TEXT NOT NULL,
                    model_version TEXT,
                    configuration_hash TEXT,
                    dependencies TEXT,
                    metadata TEXT,
                    signature TEXT NOT NULL,
                    signature_algorithm TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON provenance_records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor_id ON provenance_records(actor_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action_type ON provenance_records(action_type)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance_chains (
                    chain_id TEXT PRIMARY KEY,
                    root_record_id TEXT NOT NULL,
                    record_ids TEXT NOT NULL,  -- JSON array
                    chain_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    check_id TEXT PRIMARY KEY,
                    check_timestamp REAL NOT NULL,
                    total_records INTEGER NOT NULL,
                    verified_signatures INTEGER NOT NULL,
                    failed_signatures INTEGER NOT NULL,
                    database_hash TEXT NOT NULL
                )
            """)

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_record(self, record: ProvenanceRecord) -> bool:
        """Store a provenance record in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO provenance_records (
                        record_id, timestamp, actor_id, action_type, action_description,
                        input_data_hash, output_data_hash, model_version, configuration_hash,
                        dependencies, metadata, signature, signature_algorithm, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.timestamp.isoformat(),
                    record.actor_id,
                    record.action_type,
                    record.action_description,
                    record.input_data_hash,
                    record.output_data_hash,
                    record.model_version,
                    record.configuration_hash,
                    json.dumps(record.dependencies),
                    json.dumps(record.metadata),
                    record.signature,
                    record.signature_algorithm,
                    time.time()
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store provenance record: {e}")
            return False

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Retrieve a provenance record by ID."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM provenance_records WHERE record_id = ?",
                    (record_id,)
                ).fetchone()

                if row:
                    return ProvenanceRecord(
                        record_id=row['record_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        actor_id=row['actor_id'],
                        action_type=row['action_type'],
                        action_description=row['action_description'],
                        input_data_hash=row['input_data_hash'],
                        output_data_hash=row['output_data_hash'],
                        model_version=row['model_version'],
                        configuration_hash=row['configuration_hash'],
                        dependencies=json.loads(row['dependencies']),
                        metadata=json.loads(row['metadata']),
                        signature=row['signature'],
                        signature_algorithm=row['signature_algorithm']
                    )
        except Exception as e:
            logger.error(f"Failed to retrieve record {record_id}: {e}")

        return None

    def get_records_by_actor(self, actor_id: str, limit: int = 100) -> List[ProvenanceRecord]:
        """Get records by actor ID."""
        records = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM provenance_records
                    WHERE actor_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (actor_id, limit)).fetchall()

                for row in rows:
                    record = ProvenanceRecord(
                        record_id=row['record_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        actor_id=row['actor_id'],
                        action_type=row['action_type'],
                        action_description=row['action_description'],
                        input_data_hash=row['input_data_hash'],
                        output_data_hash=row['output_data_hash'],
                        model_version=row['model_version'],
                        configuration_hash=row['configuration_hash'],
                        dependencies=json.loads(row['dependencies']),
                        metadata=json.loads(row['metadata']),
                        signature=row['signature'],
                        signature_algorithm=row['signature_algorithm']
                    )
                    records.append(record)
        except Exception as e:
            logger.error(f"Failed to get records for actor {actor_id}: {e}")

        return records

    def store_chain(self, chain: ProvenanceChain) -> bool:
        """Store a provenance chain."""
        try:
            with self._get_connection() as conn:
                record_ids = [r.record_id for r in chain.records]
                now = time.time()

                conn.execute("""
                    INSERT OR REPLACE INTO provenance_chains (
                        chain_id, root_record_id, record_ids, chain_hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    chain.chain_id,
                    chain.root_record_id,
                    json.dumps(record_ids),
                    chain.chain_hash,
                    now,
                    now
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store provenance chain: {e}")
            return False

    def verify_database_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of all stored records."""
        signer = CryptographicSigner()

        total_records = 0
        verified_signatures = 0
        failed_signatures = 0

        try:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT * FROM provenance_records").fetchall()
                total_records = len(rows)

                for row in rows:
                    record = ProvenanceRecord(
                        record_id=row['record_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        actor_id=row['actor_id'],
                        action_type=row['action_type'],
                        action_description=row['action_description'],
                        input_data_hash=row['input_data_hash'],
                        output_data_hash=row['output_data_hash'],
                        model_version=row['model_version'],
                        configuration_hash=row['configuration_hash'],
                        dependencies=json.loads(row['dependencies']),
                        metadata=json.loads(row['metadata']),
                        signature=row['signature'],
                        signature_algorithm=row['signature_algorithm']
                    )

                    if signer.verify_signature(record):
                        verified_signatures += 1
                    else:
                        failed_signatures += 1

                # Calculate database hash
                all_data = json.dumps([dict(row) for row in rows], sort_keys=True)
                database_hash = hashlib.sha256(all_data.encode()).hexdigest()

                # Store integrity check result
                check_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO integrity_checks (
                        check_id, check_timestamp, total_records, verified_signatures,
                        failed_signatures, database_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    check_id,
                    time.time(),
                    total_records,
                    verified_signatures,
                    failed_signatures,
                    database_hash
                ))

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")

        return {
            "total_records": total_records,
            "verified_signatures": verified_signatures,
            "failed_signatures": failed_signatures,
            "integrity_percentage": (verified_signatures / total_records * 100) if total_records > 0 else 0,
            "database_hash": database_hash
        }


class ProvenanceSystem:
    """Complete provenance signing and persistence system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        self.signer = CryptographicSigner(
            self.config.get("key_dir")
        )
        self.persistence = ProvenancePersistence(
            self.config.get("db_path")
        )

        # Active chains
        self.active_chains: Dict[str, ProvenanceChain] = {}

        logger.info("Provenance system initialized with cryptographic signing")

    def record_action(
        self,
        actor_id: str,
        action_type: str,
        action_description: str,
        input_data: Any,
        output_data: Any,
        model_version: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chain_id: Optional[str] = None
    ) -> ProvenanceRecord:
        """Record a signed action in the provenance system."""

        # Generate hashes
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(output_data)

        # Create record
        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            actor_id=actor_id,
            action_type=action_type,
            action_description=action_description,
            input_data_hash=input_hash,
            output_data_hash=output_hash,
            model_version=model_version,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        # Sign the record
        record.signature = self.signer.sign_record(record)

        # Store the record
        if self.persistence.store_record(record):
            logger.info(f"Recorded signed action: {action_type} by {actor_id}")

            # Add to chain if specified
            if chain_id:
                self._add_to_chain(chain_id, record)
        else:
            logger.error(f"Failed to store provenance record: {record.record_id}")

        return record

    def start_provenance_chain(self, root_action: str) -> str:
        """Start a new provenance chain."""
        chain_id = str(uuid.uuid4())
        chain = ProvenanceChain(
            chain_id=chain_id,
            root_record_id=""  # Will be set when first record is added
        )
        self.active_chains[chain_id] = chain

        logger.info(f"Started provenance chain: {chain_id} for {root_action}")
        return chain_id

    def _add_to_chain(self, chain_id: str, record: ProvenanceRecord) -> None:
        """Add record to an active chain."""
        if chain_id in self.active_chains:
            chain = self.active_chains[chain_id]

            # Set root record if this is the first
            if not chain.root_record_id:
                chain.root_record_id = record.record_id

            chain.add_record(record)

            # Persist the updated chain
            self.persistence.store_chain(chain)

    def finalize_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """Finalize and return a provenance chain."""
        if chain_id in self.active_chains:
            chain = self.active_chains.pop(chain_id)
            logger.info(f"Finalized provenance chain: {chain_id} with {len(chain.records)} records")
            return chain
        return None

    def _hash_data(self, data: Any) -> str:
        """Generate SHA-256 hash of data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_record_integrity(self, record_id: str) -> bool:
        """Verify the cryptographic integrity of a specific record."""
        record = self.persistence.get_record(record_id)
        if record:
            return self.signer.verify_signature(record)
        return False

    def get_audit_trail(self, actor_id: str) -> List[ProvenanceRecord]:
        """Get complete audit trail for an actor."""
        return self.persistence.get_records_by_actor(actor_id)

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        integrity_check = self.persistence.verify_database_integrity()

        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "cryptographic_signatures": "enabled" if not self.signer._mock_mode else "mock_mode",
            "total_records": integrity_check["total_records"],
            "signature_verification": {
                "verified": integrity_check["verified_signatures"],
                "failed": integrity_check["failed_signatures"],
                "integrity_percentage": integrity_check["integrity_percentage"]
            },
            "database_integrity": {
                "hash": integrity_check["database_hash"],
                "tamper_proof": integrity_check["failed_signatures"] == 0
            },
            "compliance_status": "COMPLIANT" if integrity_check["integrity_percentage"] == 100 else "VIOLATIONS_DETECTED"
        }


__all__ = [
    "ProvenanceRecord",
    "ProvenanceChain",
    "ProvenanceSystem",
    "CryptographicSigner",
    "ProvenancePersistence"
]