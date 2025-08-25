"""Tree class for structured output handling."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the research tree."""
    ROOT = "root"
    QUERY = "query"
    RESULT = "result"
    INSIGHT = "insight"
    SUMMARY = "summary"
    OUTLINE = "outline"


@dataclass
class TreeNode:
    """A node in the research tree structure."""
    id: str
    type: NodeType
    content: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


class Tree:
    """
    Tree class for structured output handling in the research workflow.
    
    This class manages hierarchical data structures for organizing research
    findings, insights, and relationships between different pieces of information.
    """
    
    def __init__(self, root_content: str = "Research Session"):
        """Initialize a new research tree."""
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id = self._generate_id()
        
        # Create root node
        root_node = TreeNode(
            id=self.root_id,
            type=NodeType.ROOT,
            content=root_content,
            metadata={"created_at": self._get_timestamp()}
        )
        self.nodes[self.root_id] = root_node
    
    def add_node(
        self, 
        content: str, 
        node_type: NodeType, 
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new node to the tree."""
        node_id = self._generate_id()
        
        if parent_id is None:
            parent_id = self.root_id
        
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        # Create the new node
        new_node = TreeNode(
            id=node_id,
            type=node_type,
            content=content,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        # Add timestamp
        new_node.metadata["created_at"] = self._get_timestamp()
        
        # Add to nodes dict
        self.nodes[node_id] = new_node
        
        # Update parent's children
        self.nodes[parent_id].children_ids.append(node_id)
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[TreeNode]:
        """Get all children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
    
    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """Get the path from a node to the root."""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        
        return list(reversed(path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary representation."""
        return {
            "root_id": self.root_id,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type.value,
                    "content": node.content,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def get_insights(self) -> List[TreeNode]:
        """Get all insight nodes from the tree."""
        return [node for node in self.nodes.values() if node.type == NodeType.INSIGHT]
    
    def extract_insights(self) -> List[str]:
        """Extract insights as a list of strings."""
        insight_nodes = self.get_insights()
        return [node.content for node in insight_nodes]
    
    def get_results(self) -> List[TreeNode]:
        """Get all result nodes from the tree."""
        return [node for node in self.nodes.values() if node.type == NodeType.RESULT]
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a node."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def __repr__(self) -> str:
        """String representation of the tree."""
        return f"Tree(nodes={len(self.nodes)}, root='{self.nodes[self.root_id].content}')"
