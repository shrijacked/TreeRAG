"""Project-wide exception types."""


class TreeRAGError(Exception):
    """Base exception for TreeRAG failures."""


class ConfigurationError(TreeRAGError):
    """Raised when user configuration is invalid."""


class ProviderError(TreeRAGError):
    """Raised when the LLM provider cannot complete a request."""


class ParseError(TreeRAGError):
    """Raised when segmentation or index construction fails."""


class RoutingError(TreeRAGError):
    """Raised when tree navigation cannot pick a valid child node."""


class StorageError(TreeRAGError):
    """Raised when index serialization or deserialization fails."""
