"""
Graph Neural Network System for Resource Analysis

This module implements a comprehensive graph neural network system for analyzing
cloud resource relationships, dependencies, and optimization opportunities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from decimal import Decimal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx

from .models import BaseModel
from .exceptions import FinOpsException

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of cloud resources"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    SECURITY_GROUP = "security_group"
    VPC = "vpc"
    SUBNET = "subnet"
    APPLICATION = "application"
    SERVICE = "service"


class RelationshipType(Enum):
    """Types of relationships between resources"""
    DEPENDS_ON = "depends_on"
    CONNECTS_TO = "connects_to"
    CONTAINS = "contains"
    USES = "uses"
    COMMUNICATES_WITH = "communicates_with"
    SHARES_NETWORK = "shares_network"
    BACKUP_OF = "backup_of"
    REPLICA_OF = "replica_of"


class OptimizationOpportunityType(Enum):
    """Types of optimization opportunities"""
    RIGHTSIZING = "rightsizing"
    CONSOLIDATION = "consolidation"
    MIGRATION = "migration"
    SCALING = "scaling"
    DECOMMISSION = "decommission"
    NETWORK_OPTIMIZATION = "network_optimization"


@dataclass
class ResourceNode:
    """Represents a cloud resource node in the graph"""
    resource_id: str
    resource_type: ResourceType
    provider: str
    region: str
    properties: Dict[str, Any] = field(default_factory=dict)
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type.value,
            'provider': self.provider,
            'region': self.region,
            'properties': self.properties,
            'cost_metrics': self.cost_metrics,
            'performance_metrics': self.performance_metrics,
            'embeddings': self.embeddings
        }


@dataclass
class ResourceEdge:
    """Represents a relationship between resources"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'properties': self.properties
        }


@dataclass
class ResourceGraph:
    """Complete resource graph representation"""
    nodes: List[ResourceNode]
    edges: List[ResourceEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }


@dataclass
class OptimizationOpportunity:
    """Represents an optimization opportunity"""
    opportunity_id: str
    opportunity_type: OptimizationOpportunityType
    affected_resources: List[str]
    estimated_savings: Decimal
    confidence_score: float
    description: str
    implementation_complexity: str
    risk_level: str
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'opportunity_id': self.opportunity_id,
            'opportunity_type': self.opportunity_type.value,
            'affected_resources': self.affected_resources,
            'estimated_savings': float(self.estimated_savings),
            'confidence_score': self.confidence_score,
            'description': self.description,
            'implementation_complexity': self.implementation_complexity,
            'risk_level': self.risk_level,
            'prerequisites': self.prerequisites
        }


@dataclass
class DependencyAnalysis:
    """Results of dependency analysis"""
    critical_paths: List[List[str]]
    bottlenecks: List[str]
    optimization_opportunities: List[OptimizationOpportunity]
    risk_assessment: Dict[str, Any]
    dependency_matrix: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'critical_paths': self.critical_paths,
            'bottlenecks': self.bottlenecks,
            'optimization_opportunities': [opp.to_dict() for opp in self.optimization_opportunities],
            'risk_assessment': self.risk_assessment,
            'dependency_matrix': self.dependency_matrix
        }


@dataclass
class CascadeEffects:
    """Predicted cascade effects of optimization actions"""
    primary_impact: Dict[str, Any]
    secondary_impacts: List[Dict[str, Any]]
    risk_score: float
    affected_resources: List[str]
    mitigation_strategies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'primary_impact': self.primary_impact,
            'secondary_impacts': self.secondary_impacts,
            'risk_score': self.risk_score,
            'affected_resources': self.affected_resources,
            'mitigation_strategies': self.mitigation_strategies
        }


@dataclass
class ResourceCluster:
    """Group of related resources"""
    cluster_id: str
    resources: List[str]
    cluster_type: str
    cohesion_score: float
    optimization_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cluster_id': self.cluster_id,
            'resources': self.resources,
            'cluster_type': self.cluster_type,
            'cohesion_score': self.cohesion_score,
            'optimization_potential': self.optimization_potential
        }


@dataclass
class ClusterOptimization:
    """Optimization recommendations for a resource cluster"""
    cluster_id: str
    recommendations: List[OptimizationOpportunity]
    coordination_strategy: str
    implementation_order: List[str]
    total_estimated_savings: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cluster_id': self.cluster_id,
            'recommendations': [rec.to_dict() for rec in self.recommendations],
            'coordination_strategy': self.coordination_strategy,
            'implementation_order': self.implementation_order,
            'total_estimated_savings': float(self.total_estimated_savings)
        }


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for resource analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Attention mechanism for important relationships
        self.attention = GATConv(output_dim, output_dim, heads=4, concat=False)
        
        # Classification heads for different tasks
        self.bottleneck_classifier = nn.Linear(output_dim, 2)  # Binary: bottleneck or not
        self.optimization_classifier = nn.Linear(output_dim, len(OptimizationOpportunityType))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the network"""
        # Graph convolutions with residual connections
        h = x
        for i, conv in enumerate(self.convs):
            h_new = F.relu(conv(h, edge_index))
            h_new = self.dropout(h_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Apply attention
        h = self.attention(h, edge_index)
        
        # Global pooling for graph-level predictions
        if batch is not None:
            graph_embedding = global_mean_pool(h, batch)
        else:
            graph_embedding = torch.mean(h, dim=0, keepdim=True)
        
        # Predictions
        bottleneck_pred = self.bottleneck_classifier(h)
        optimization_pred = self.optimization_classifier(h)
        
        return {
            'node_embeddings': h,
            'graph_embedding': graph_embedding,
            'bottleneck_predictions': bottleneck_pred,
            'optimization_predictions': optimization_pred
        }


class ResourceGraphBuilder:
    """Builds graph representations of cloud infrastructure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ResourceGraphBuilder")
    
    async def build_graph_from_resources(self, resources: List[Dict[str, Any]]) -> ResourceGraph:
        """Build a resource graph from a list of resources"""
        try:
            nodes = []
            edges = []
            
            # Create nodes
            for resource in resources:
                node = ResourceNode(
                    resource_id=resource['id'],
                    resource_type=ResourceType(resource.get('type', 'compute')),
                    provider=resource.get('provider', 'aws'),
                    region=resource.get('region', 'us-east-1'),
                    properties=resource.get('properties', {}),
                    cost_metrics=resource.get('cost_metrics', {}),
                    performance_metrics=resource.get('performance_metrics', {})
                )
                nodes.append(node)
            
            # Infer relationships between resources
            edges = await self._infer_relationships(nodes)
            
            # Create metadata
            metadata = {
                'created_at': datetime.utcnow().isoformat(),
                'node_count': len(nodes),
                'edge_count': len(edges),
                'providers': list(set(node.provider for node in nodes)),
                'regions': list(set(node.region for node in nodes))
            }
            
            return ResourceGraph(nodes=nodes, edges=edges, metadata=metadata)
            
        except Exception as e:
            self.logger.error(f"Error building resource graph: {str(e)}")
            raise FinOpsException(f"Failed to build resource graph: {str(e)}")
    
    async def _infer_relationships(self, nodes: List[ResourceNode]) -> List[ResourceEdge]:
        """Infer relationships between resources based on their properties"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:  # Avoid duplicate edges and self-loops
                    continue
                
                # Check for various relationship types
                relationships = await self._detect_relationships(node1, node2)
                edges.extend(relationships)
        
        return edges
    
    async def _detect_relationships(self, node1: ResourceNode, node2: ResourceNode) -> List[ResourceEdge]:
        """Detect relationships between two nodes"""
        relationships = []
        
        # Same region relationship (weaker)
        if node1.region == node2.region:
            relationships.append(ResourceEdge(
                source_id=node1.resource_id,
                target_id=node2.resource_id,
                relationship_type=RelationshipType.SHARES_NETWORK,
                strength=0.3,
                properties={'reason': 'same_region'}
            ))
        
        # VPC containment
        if (node1.resource_type == ResourceType.VPC and 
            node2.properties.get('vpc_id') == node1.resource_id):
            relationships.append(ResourceEdge(
                source_id=node1.resource_id,
                target_id=node2.resource_id,
                relationship_type=RelationshipType.CONTAINS,
                strength=0.9,
                properties={'reason': 'vpc_containment'}
            ))
        
        # Load balancer to compute
        if (node1.resource_type == ResourceType.LOAD_BALANCER and
            node2.resource_type == ResourceType.COMPUTE):
            if node2.resource_id in node1.properties.get('target_instances', []):
                relationships.append(ResourceEdge(
                    source_id=node1.resource_id,
                    target_id=node2.resource_id,
                    relationship_type=RelationshipType.CONNECTS_TO,
                    strength=0.8,
                    properties={'reason': 'load_balancer_target'}
                ))
        
        # Database dependencies
        if (node1.resource_type == ResourceType.DATABASE and
            node2.resource_type in [ResourceType.COMPUTE, ResourceType.APPLICATION]):
            if node1.resource_id in node2.properties.get('database_connections', []):
                relationships.append(ResourceEdge(
                    source_id=node2.resource_id,
                    target_id=node1.resource_id,
                    relationship_type=RelationshipType.DEPENDS_ON,
                    strength=0.9,
                    properties={'reason': 'database_dependency'}
                ))
        
        # Storage attachments
        if (node1.resource_type == ResourceType.STORAGE and
            node2.resource_type == ResourceType.COMPUTE):
            if node1.resource_id in node2.properties.get('attached_volumes', []):
                relationships.append(ResourceEdge(
                    source_id=node2.resource_id,
                    target_id=node1.resource_id,
                    relationship_type=RelationshipType.USES,
                    strength=0.7,
                    properties={'reason': 'storage_attachment'}
                ))
        
        return relationships
    
    def to_pytorch_geometric(self, graph: ResourceGraph) -> Data:
        """Convert ResourceGraph to PyTorch Geometric Data object"""
        # Create node feature matrix
        node_features = []
        node_id_to_index = {}
        
        for i, node in enumerate(graph.nodes):
            node_id_to_index[node.resource_id] = i
            
            # Create feature vector for node
            features = self._create_node_features(node)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge index
        edge_indices = []
        edge_attrs = []
        
        for edge in graph.edges:
            if edge.source_id in node_id_to_index and edge.target_id in node_id_to_index:
                source_idx = node_id_to_index[edge.source_id]
                target_idx = node_id_to_index[edge.target_id]
                
                # Add both directions for undirected graph
                edge_indices.extend([[source_idx, target_idx], [target_idx, source_idx]])
                
                # Edge attributes
                edge_attr = [edge.strength, len(RelationshipType)]  # Basic edge features
                edge_attrs.extend([edge_attr, edge_attr])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _create_node_features(self, node: ResourceNode) -> List[float]:
        """Create feature vector for a node"""
        features = []
        
        # Resource type one-hot encoding
        resource_type_features = [0.0] * len(ResourceType)
        resource_type_features[list(ResourceType).index(node.resource_type)] = 1.0
        features.extend(resource_type_features)
        
        # Cost metrics (normalized)
        cost_features = [
            node.cost_metrics.get('monthly_cost', 0.0) / 1000.0,  # Normalize by $1000
            node.cost_metrics.get('hourly_cost', 0.0) * 24 * 30 / 1000.0,  # Monthly equivalent
            node.cost_metrics.get('utilization', 0.0) / 100.0  # Normalize percentage
        ]
        features.extend(cost_features)
        
        # Performance metrics (normalized)
        perf_features = [
            node.performance_metrics.get('cpu_utilization', 0.0) / 100.0,
            node.performance_metrics.get('memory_utilization', 0.0) / 100.0,
            node.performance_metrics.get('network_utilization', 0.0) / 100.0
        ]
        features.extend(perf_features)
        
        # Additional features
        additional_features = [
            1.0 if node.provider == 'aws' else 0.0,
            1.0 if node.provider == 'gcp' else 0.0,
            1.0 if node.provider == 'azure' else 0.0,
            len(node.properties) / 10.0  # Normalize property count
        ]
        features.extend(additional_features)
        
        return features


class DependencyAnalyzer:
    """Analyzes resource dependencies and identifies critical paths"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".DependencyAnalyzer")
    
    async def analyze_dependencies(self, graph: ResourceGraph) -> DependencyAnalysis:
        """Perform comprehensive dependency analysis"""
        try:
            # Convert to NetworkX for analysis
            nx_graph = self._to_networkx(graph)
            
            # Find critical paths
            critical_paths = await self._find_critical_paths(nx_graph)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(nx_graph, graph)
            
            # Find optimization opportunities
            opportunities = await self._find_optimization_opportunities(graph, nx_graph)
            
            # Assess risks
            risk_assessment = await self._assess_risks(graph, critical_paths, bottlenecks)
            
            # Create dependency matrix
            dependency_matrix = await self._create_dependency_matrix(nx_graph)
            
            return DependencyAnalysis(
                critical_paths=critical_paths,
                bottlenecks=bottlenecks,
                optimization_opportunities=opportunities,
                risk_assessment=risk_assessment,
                dependency_matrix=dependency_matrix
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {str(e)}")
            raise FinOpsException(f"Failed to analyze dependencies: {str(e)}")
    
    def _to_networkx(self, graph: ResourceGraph) -> nx.DiGraph:
        """Convert ResourceGraph to NetworkX directed graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph.nodes:
            G.add_node(node.resource_id, **node.to_dict())
        
        # Add edges
        for edge in graph.edges:
            G.add_edge(
                edge.source_id, 
                edge.target_id, 
                weight=edge.strength,
                **edge.to_dict()
            )
        
        return G
    
    async def _find_critical_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find critical paths in the dependency graph"""
        critical_paths = []
        
        try:
            # Method 1: Find paths from sources to sinks
            sources = [node for node in graph.nodes() if graph.in_degree(node) == 0]
            sinks = [node for node in graph.nodes() if graph.out_degree(node) == 0]
            
            # If no clear sources/sinks, find nodes with high in-degree (many depend on them)
            if not sources:
                # Find nodes that many others depend on
                high_indegree_nodes = sorted(
                    [(node, graph.in_degree(node)) for node in graph.nodes()],
                    key=lambda x: x[1], reverse=True
                )[:3]
                sources = [node for node, _ in high_indegree_nodes if graph.in_degree(node) > 0]
            
            if not sinks:
                # Find nodes that depend on many others
                high_outdegree_nodes = sorted(
                    [(node, graph.out_degree(node)) for node in graph.nodes()],
                    key=lambda x: x[1], reverse=True
                )[:3]
                sinks = [node for node, _ in high_outdegree_nodes if graph.out_degree(node) > 0]
            
            # Find paths between sources and sinks
            for source in sources[:3]:  # Limit sources
                for sink in sinks[:3]:  # Limit sinks
                    if source != sink:
                        try:
                            if nx.has_path(graph, source, sink):
                                path = nx.shortest_path(graph, source, sink)
                                if len(path) >= 2:  # At least 2 nodes
                                    critical_paths.append(path)
                        except (nx.NetworkXNoPath, nx.NetworkXError):
                            continue
            
            # Method 2: Find dependency chains (A depends on B depends on C)
            dependency_chains = []
            
            # Look for chains of DEPENDS_ON relationships
            for node in graph.nodes():
                chain = [node]
                current = node
                
                # Follow outgoing DEPENDS_ON edges
                while True:
                    dependencies = [
                        target for target in graph.successors(current)
                        if graph[current][target].get('relationship_type') == 'depends_on'
                    ]
                    
                    if dependencies and len(chain) < 5:  # Limit chain length
                        next_node = dependencies[0]  # Take first dependency
                        if next_node not in chain:  # Avoid cycles
                            chain.append(next_node)
                            current = next_node
                        else:
                            break
                    else:
                        break
                
                if len(chain) >= 2:
                    dependency_chains.append(chain)
            
            # Add dependency chains to critical paths
            critical_paths.extend(dependency_chains)
            
            # Method 3: Find paths between high-centrality nodes
            try:
                centrality = nx.degree_centrality(graph)
                high_centrality_nodes = sorted(
                    centrality.items(), key=lambda x: x[1], reverse=True
                )[:5]  # Top 5 nodes by centrality
                
                for i, (source, _) in enumerate(high_centrality_nodes):
                    for target, _ in high_centrality_nodes[i+1:]:
                        try:
                            if nx.has_path(graph, source, target):
                                path = nx.shortest_path(graph, source, target)
                                if len(path) >= 2:
                                    critical_paths.append(path)
                            elif nx.has_path(graph, target, source):
                                path = nx.shortest_path(graph, target, source)
                                if len(path) >= 2:
                                    critical_paths.append(path)
                        except (nx.NetworkXNoPath, nx.NetworkXError):
                            continue
            except Exception as e:
                self.logger.warning(f"Error in centrality-based path finding: {e}")
            
            # Remove duplicates and sort by length (longest first)
            unique_paths = []
            for path in critical_paths:
                if path not in unique_paths and len(path) >= 2:
                    unique_paths.append(path)
            
            # Sort by length (longest first) and importance
            try:
                centrality = nx.degree_centrality(graph)
                unique_paths.sort(
                    key=lambda x: (len(x), sum(centrality.get(node, 0) for node in x)), 
                    reverse=True
                )
            except:
                unique_paths.sort(key=len, reverse=True)
            
            return unique_paths[:10]  # Return top 10 critical paths
            
        except Exception as e:
            self.logger.warning(f"Error finding critical paths: {str(e)}")
        
        return critical_paths[:10]  # Limit to top 10 critical paths
    
    async def _identify_bottlenecks(self, graph: nx.DiGraph, resource_graph: ResourceGraph) -> List[str]:
        """Identify bottleneck resources"""
        bottlenecks = []
        
        try:
            # Calculate centrality measures
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            degree_centrality = nx.degree_centrality(graph)
            
            # Combine centrality measures
            combined_centrality = {}
            for node in graph.nodes():
                combined_centrality[node] = (
                    0.4 * betweenness.get(node, 0) +
                    0.3 * closeness.get(node, 0) +
                    0.3 * degree_centrality.get(node, 0)
                )
            
            # Sort by centrality and take top candidates
            sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Filter based on resource characteristics
            for node_id, centrality in sorted_nodes[:20]:  # Top 20 candidates
                node = next((n for n in resource_graph.nodes if n.resource_id == node_id), None)
                if node and self._is_bottleneck_candidate(node, centrality):
                    bottlenecks.append(node_id)
                
                if len(bottlenecks) >= 10:  # Limit to top 10 bottlenecks
                    break
            
        except Exception as e:
            self.logger.warning(f"Error identifying bottlenecks: {str(e)}")
        
        return bottlenecks
    
    def _is_bottleneck_candidate(self, node: ResourceNode, centrality: float) -> bool:
        """Determine if a node is a bottleneck candidate"""
        # High centrality threshold
        if centrality < 0.1:
            return False
        
        # Check resource utilization
        cpu_util = node.performance_metrics.get('cpu_utilization', 0)
        memory_util = node.performance_metrics.get('memory_utilization', 0)
        network_util = node.performance_metrics.get('network_utilization', 0)
        
        # High utilization indicates potential bottleneck
        if cpu_util > 80 or memory_util > 80 or network_util > 80:
            return True
        
        # Critical resource types
        if node.resource_type in [ResourceType.DATABASE, ResourceType.LOAD_BALANCER]:
            return True
        
        return False
    
    async def _find_optimization_opportunities(self, graph: ResourceGraph, nx_graph: nx.DiGraph) -> List[OptimizationOpportunity]:
        """Find optimization opportunities in the graph"""
        opportunities = []
        
        try:
            # Find underutilized resources
            for node in graph.nodes:
                cpu_util = node.performance_metrics.get('cpu_utilization', 0)
                memory_util = node.performance_metrics.get('memory_utilization', 0)
                
                if cpu_util < 20 and memory_util < 30:
                    opportunities.append(OptimizationOpportunity(
                        opportunity_id=f"rightsizing_{node.resource_id}",
                        opportunity_type=OptimizationOpportunityType.RIGHTSIZING,
                        affected_resources=[node.resource_id],
                        estimated_savings=Decimal(str(node.cost_metrics.get('monthly_cost', 0) * 0.3)),
                        confidence_score=0.8,
                        description=f"Rightsize underutilized {node.resource_type.value}",
                        implementation_complexity="low",
                        risk_level="low"
                    ))
            
            # Find consolidation opportunities
            consolidation_groups = await self._find_consolidation_groups(graph, nx_graph)
            for group in consolidation_groups:
                total_cost = sum(
                    node.cost_metrics.get('monthly_cost', 0) 
                    for node in graph.nodes 
                    if node.resource_id in group
                )
                opportunities.append(OptimizationOpportunity(
                    opportunity_id=f"consolidation_{hash(tuple(group))}",
                    opportunity_type=OptimizationOpportunityType.CONSOLIDATION,
                    affected_resources=group,
                    estimated_savings=Decimal(str(total_cost * 0.25)),
                    confidence_score=0.6,
                    description=f"Consolidate {len(group)} related resources",
                    implementation_complexity="medium",
                    risk_level="medium"
                ))
            
        except Exception as e:
            self.logger.warning(f"Error finding optimization opportunities: {str(e)}")
        
        return opportunities[:20]  # Limit to top 20 opportunities
    
    async def _find_consolidation_groups(self, graph: ResourceGraph, nx_graph: nx.DiGraph) -> List[List[str]]:
        """Find groups of resources that can be consolidated"""
        groups = []
        
        try:
            # Find communities/clusters in the graph
            undirected_graph = nx_graph.to_undirected()
            
            # Use simple clustering based on connectivity
            visited = set()
            for node in undirected_graph.nodes():
                if node not in visited:
                    # Find connected component
                    component = nx.node_connected_component(undirected_graph, node)
                    
                    # Filter for consolidation candidates
                    candidates = []
                    for node_id in component:
                        node_obj = next((n for n in graph.nodes if n.resource_id == node_id), None)
                        if (node_obj and 
                            node_obj.resource_type == ResourceType.COMPUTE and
                            node_obj.performance_metrics.get('cpu_utilization', 0) < 50):
                            candidates.append(node_id)
                    
                    if len(candidates) >= 2:
                        groups.append(candidates)
                    
                    visited.update(component)
            
        except Exception as e:
            self.logger.warning(f"Error finding consolidation groups: {str(e)}")
        
        return groups
    
    async def _assess_risks(self, graph: ResourceGraph, critical_paths: List[List[str]], bottlenecks: List[str]) -> Dict[str, Any]:
        """Assess risks in the resource graph"""
        risk_assessment = {
            'overall_risk_score': 0.0,
            'single_points_of_failure': [],
            'high_dependency_resources': [],
            'cross_region_dependencies': 0,
            'cross_provider_dependencies': 0,
            'recommendations': []
        }
        
        try:
            # Single points of failure
            spofs = []
            for node in graph.nodes:
                # Count incoming dependencies
                incoming_deps = sum(1 for edge in graph.edges if edge.target_id == node.resource_id)
                if incoming_deps > 5:  # High dependency count
                    spofs.append(node.resource_id)
            
            risk_assessment['single_points_of_failure'] = spofs
            risk_assessment['high_dependency_resources'] = bottlenecks
            
            # Cross-region/provider dependencies
            cross_region = 0
            cross_provider = 0
            
            for edge in graph.edges:
                source_node = next((n for n in graph.nodes if n.resource_id == edge.source_id), None)
                target_node = next((n for n in graph.nodes if n.resource_id == edge.target_id), None)
                
                if source_node and target_node:
                    if source_node.region != target_node.region:
                        cross_region += 1
                    if source_node.provider != target_node.provider:
                        cross_provider += 1
            
            risk_assessment['cross_region_dependencies'] = cross_region
            risk_assessment['cross_provider_dependencies'] = cross_provider
            
            # Calculate overall risk score
            risk_score = (
                len(spofs) * 0.3 +
                len(bottlenecks) * 0.2 +
                len(critical_paths) * 0.2 +
                cross_region * 0.15 +
                cross_provider * 0.15
            ) / max(len(graph.nodes), 1)
            
            risk_assessment['overall_risk_score'] = min(risk_score, 1.0)
            
            # Generate recommendations
            recommendations = []
            if len(spofs) > 0:
                recommendations.append("Consider adding redundancy for high-dependency resources")
            if cross_region > len(graph.nodes) * 0.2:
                recommendations.append("Review cross-region dependencies for latency and cost impact")
            if len(bottlenecks) > 0:
                recommendations.append("Monitor and potentially scale bottleneck resources")
            
            risk_assessment['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.warning(f"Error assessing risks: {str(e)}")
        
        return risk_assessment
    
    async def _create_dependency_matrix(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Create a dependency matrix showing direct dependencies"""
        matrix = {}
        
        for node in graph.nodes():
            dependencies = list(graph.successors(node))
            matrix[node] = dependencies
        
        return matrix


class CascadePredictor:
    """Predicts cascade effects of optimization actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".CascadePredictor")
    
    async def predict_cascade_effects(self, action: Dict[str, Any], graph: ResourceGraph) -> CascadeEffects:
        """Predict cascade effects of an optimization action"""
        try:
            # Convert to NetworkX for analysis
            nx_graph = self._to_networkx(graph)
            
            # Identify primary impact
            primary_impact = await self._analyze_primary_impact(action, graph)
            
            # Identify secondary impacts
            secondary_impacts = await self._analyze_secondary_impacts(action, graph, nx_graph)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(primary_impact, secondary_impacts)
            
            # Find all affected resources
            affected_resources = await self._find_affected_resources(action, secondary_impacts, nx_graph)
            
            # Generate mitigation strategies
            mitigation_strategies = await self._generate_mitigation_strategies(
                action, primary_impact, secondary_impacts
            )
            
            return CascadeEffects(
                primary_impact=primary_impact,
                secondary_impacts=secondary_impacts,
                risk_score=risk_score,
                affected_resources=affected_resources,
                mitigation_strategies=mitigation_strategies
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting cascade effects: {str(e)}")
            raise FinOpsException(f"Failed to predict cascade effects: {str(e)}")
    
    def _to_networkx(self, graph: ResourceGraph) -> nx.DiGraph:
        """Convert ResourceGraph to NetworkX directed graph"""
        G = nx.DiGraph()
        
        for node in graph.nodes:
            G.add_node(node.resource_id, **node.to_dict())
        
        for edge in graph.edges:
            G.add_edge(edge.source_id, edge.target_id, weight=edge.strength, **edge.to_dict())
        
        return G
    
    async def _analyze_primary_impact(self, action: Dict[str, Any], graph: ResourceGraph) -> Dict[str, Any]:
        """Analyze the primary impact of an action"""
        target_resource_id = action.get('target_resource_id')
        action_type = action.get('action_type', 'unknown')
        
        # Find the target resource
        target_resource = next(
            (node for node in graph.nodes if node.resource_id == target_resource_id), 
            None
        )
        
        if not target_resource:
            return {'error': 'Target resource not found'}
        
        primary_impact = {
            'resource_id': target_resource_id,
            'action_type': action_type,
            'resource_type': target_resource.resource_type.value,
            'estimated_cost_change': 0.0,
            'estimated_performance_change': {},
            'availability_impact': 'none'
        }
        
        # Estimate impacts based on action type
        if action_type == 'rightsizing':
            scale_factor = action.get('scale_factor', 0.5)
            current_cost = target_resource.cost_metrics.get('monthly_cost', 0)
            primary_impact['estimated_cost_change'] = current_cost * (scale_factor - 1)
            primary_impact['estimated_performance_change'] = {
                'cpu_capacity': scale_factor,
                'memory_capacity': scale_factor
            }
            primary_impact['availability_impact'] = 'temporary' if scale_factor < 1 else 'none'
        
        elif action_type == 'decommission':
            primary_impact['estimated_cost_change'] = -target_resource.cost_metrics.get('monthly_cost', 0)
            primary_impact['availability_impact'] = 'permanent'
        
        elif action_type == 'migration':
            # Migration typically has temporary availability impact
            primary_impact['availability_impact'] = 'temporary'
            primary_impact['estimated_cost_change'] = action.get('cost_change', 0)
        
        return primary_impact
    
    async def _analyze_secondary_impacts(self, action: Dict[str, Any], graph: ResourceGraph, nx_graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Analyze secondary impacts on dependent resources"""
        secondary_impacts = []
        target_resource_id = action.get('target_resource_id')
        
        try:
            # Find resources that depend on the target
            dependent_resources = list(nx_graph.predecessors(target_resource_id))
            
            # Find resources that the target depends on
            dependency_resources = list(nx_graph.successors(target_resource_id))
            
            # Analyze impact on dependent resources
            for dep_id in dependent_resources:
                dep_resource = next((node for node in graph.nodes if node.resource_id == dep_id), None)
                if dep_resource:
                    impact = await self._calculate_dependency_impact(action, dep_resource, 'dependent')
                    secondary_impacts.append(impact)
            
            # Analyze impact on dependency resources
            for dep_id in dependency_resources:
                dep_resource = next((node for node in graph.nodes if node.resource_id == dep_id), None)
                if dep_resource:
                    impact = await self._calculate_dependency_impact(action, dep_resource, 'dependency')
                    secondary_impacts.append(impact)
            
            # Find resources in the same cluster/region that might be affected
            target_resource = next((node for node in graph.nodes if node.resource_id == target_resource_id), None)
            if target_resource:
                cluster_impacts = await self._analyze_cluster_impacts(action, target_resource, graph)
                secondary_impacts.extend(cluster_impacts)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing secondary impacts: {str(e)}")
        
        return secondary_impacts
    
    async def _calculate_dependency_impact(self, action: Dict[str, Any], resource: ResourceNode, relationship: str) -> Dict[str, Any]:
        """Calculate impact on a dependent or dependency resource"""
        impact = {
            'resource_id': resource.resource_id,
            'resource_type': resource.resource_type.value,
            'relationship': relationship,
            'impact_severity': 'low',
            'estimated_cost_change': 0.0,
            'performance_impact': {},
            'mitigation_required': False
        }
        
        action_type = action.get('action_type', 'unknown')
        
        if action_type == 'decommission':
            if relationship == 'dependent':
                impact['impact_severity'] = 'high'
                impact['mitigation_required'] = True
                impact['performance_impact'] = {'availability': 'degraded'}
            elif relationship == 'dependency':
                impact['impact_severity'] = 'medium'
                impact['performance_impact'] = {'load_increase': 'possible'}
        
        elif action_type == 'rightsizing':
            scale_factor = action.get('scale_factor', 0.5)
            if scale_factor < 1 and relationship == 'dependent':
                impact['impact_severity'] = 'medium'
                impact['performance_impact'] = {'response_time': 'increased'}
        
        elif action_type == 'migration':
            impact['impact_severity'] = 'medium'
            impact['performance_impact'] = {'latency': 'temporary_increase'}
        
        return impact
    
    async def _analyze_cluster_impacts(self, action: Dict[str, Any], target_resource: ResourceNode, graph: ResourceGraph) -> List[Dict[str, Any]]:
        """Analyze impacts on resources in the same cluster/region"""
        cluster_impacts = []
        
        # Find resources in the same region and provider
        cluster_resources = [
            node for node in graph.nodes
            if (node.region == target_resource.region and 
                node.provider == target_resource.provider and
                node.resource_id != target_resource.resource_id)
        ]
        
        action_type = action.get('action_type', 'unknown')
        
        for resource in cluster_resources[:5]:  # Limit to 5 cluster resources
            impact = {
                'resource_id': resource.resource_id,
                'resource_type': resource.resource_type.value,
                'relationship': 'cluster_member',
                'impact_severity': 'low',
                'estimated_cost_change': 0.0,
                'performance_impact': {},
                'mitigation_required': False
            }
            
            # Network-related resources might be affected by migrations
            if (action_type == 'migration' and 
                resource.resource_type in [ResourceType.NETWORK, ResourceType.LOAD_BALANCER]):
                impact['impact_severity'] = 'medium'
                impact['performance_impact'] = {'network_reconfiguration': 'required'}
            
            cluster_impacts.append(impact)
        
        return cluster_impacts
    
    async def _calculate_risk_score(self, primary_impact: Dict[str, Any], secondary_impacts: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score for the cascade effects"""
        risk_score = 0.0
        
        # Primary impact risk
        availability_impact = primary_impact.get('availability_impact', 'none')
        if availability_impact == 'permanent':
            risk_score += 0.4
        elif availability_impact == 'temporary':
            risk_score += 0.2
        
        # Secondary impact risk
        high_severity_count = sum(1 for impact in secondary_impacts if impact.get('impact_severity') == 'high')
        medium_severity_count = sum(1 for impact in secondary_impacts if impact.get('impact_severity') == 'medium')
        
        risk_score += high_severity_count * 0.3
        risk_score += medium_severity_count * 0.1
        
        # Mitigation requirement risk
        mitigation_required_count = sum(1 for impact in secondary_impacts if impact.get('mitigation_required', False))
        risk_score += mitigation_required_count * 0.1
        
        return min(risk_score, 1.0)
    
    async def _find_affected_resources(self, action: Dict[str, Any], secondary_impacts: List[Dict[str, Any]], nx_graph: nx.DiGraph) -> List[str]:
        """Find all resources affected by the action"""
        affected = [action.get('target_resource_id')]
        
        for impact in secondary_impacts:
            resource_id = impact.get('resource_id')
            if resource_id and resource_id not in affected:
                affected.append(resource_id)
        
        return affected
    
    async def _generate_mitigation_strategies(self, action: Dict[str, Any], primary_impact: Dict[str, Any], secondary_impacts: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies for the cascade effects"""
        strategies = []
        
        action_type = action.get('action_type', 'unknown')
        
        # General strategies
        if primary_impact.get('availability_impact') != 'none':
            strategies.append("Schedule action during maintenance window")
            strategies.append("Implement gradual rollout with monitoring")
        
        # Action-specific strategies
        if action_type == 'decommission':
            mitigation_required = any(impact.get('mitigation_required', False) for impact in secondary_impacts)
            if mitigation_required:
                strategies.append("Migrate dependent workloads before decommissioning")
                strategies.append("Update load balancer configurations")
        
        elif action_type == 'rightsizing':
            strategies.append("Monitor performance metrics during and after resize")
            strategies.append("Have rollback plan ready")
        
        elif action_type == 'migration':
            strategies.append("Test connectivity from new location")
            strategies.append("Update DNS and routing configurations")
            strategies.append("Validate data synchronization")
        
        # High-impact mitigation
        high_impact_count = sum(1 for impact in secondary_impacts if impact.get('impact_severity') == 'high')
        if high_impact_count > 0:
            strategies.append("Conduct thorough impact assessment")
            strategies.append("Prepare emergency rollback procedures")
        
        return strategies


class ClusterOptimizer:
    """Optimizes groups of related resources together"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ClusterOptimizer")
    
    async def optimize_resource_cluster(self, cluster: ResourceCluster, graph: ResourceGraph) -> ClusterOptimization:
        """Optimize a cluster of related resources"""
        try:
            # Get cluster resources
            cluster_resources = [
                node for node in graph.nodes 
                if node.resource_id in cluster.resources
            ]
            
            # Generate optimization recommendations
            recommendations = await self._generate_cluster_recommendations(cluster_resources, graph)
            
            # Determine coordination strategy
            coordination_strategy = await self._determine_coordination_strategy(cluster, recommendations)
            
            # Plan implementation order
            implementation_order = await self._plan_implementation_order(recommendations, cluster_resources)
            
            # Calculate total savings
            total_savings = sum(rec.estimated_savings for rec in recommendations)
            
            return ClusterOptimization(
                cluster_id=cluster.cluster_id,
                recommendations=recommendations,
                coordination_strategy=coordination_strategy,
                implementation_order=implementation_order,
                total_estimated_savings=total_savings
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing resource cluster: {str(e)}")
            raise FinOpsException(f"Failed to optimize resource cluster: {str(e)}")
    
    async def _generate_cluster_recommendations(self, resources: List[ResourceNode], graph: ResourceGraph) -> List[OptimizationOpportunity]:
        """Generate optimization recommendations for cluster resources"""
        recommendations = []
        
        # Individual resource optimization (rightsizing)
        for resource in resources:
            cpu_util = resource.performance_metrics.get('cpu_utilization', 0)
            memory_util = resource.performance_metrics.get('memory_utilization', 0)
            storage_util = resource.performance_metrics.get('storage_utilization', 0)
            cost = resource.cost_metrics.get('monthly_cost', 0)
            
            # Rightsizing opportunities for underutilized resources
            if cpu_util < 30 and memory_util < 40 and cost > 0:
                recommendations.append(OptimizationOpportunity(
                    opportunity_id=f"rightsize_{resource.resource_id}",
                    opportunity_type=OptimizationOpportunityType.RIGHTSIZING,
                    affected_resources=[resource.resource_id],
                    estimated_savings=Decimal(str(cost * 0.25)),  # 25% savings from rightsizing
                    confidence_score=0.8,
                    description=f"Rightsize underutilized {resource.resource_type.value}",
                    implementation_complexity="low",
                    risk_level="low"
                ))
            
            # Storage optimization
            if resource.resource_type == ResourceType.STORAGE and storage_util < 50 and cost > 0:
                recommendations.append(OptimizationOpportunity(
                    opportunity_id=f"storage_opt_{resource.resource_id}",
                    opportunity_type=OptimizationOpportunityType.RIGHTSIZING,
                    affected_resources=[resource.resource_id],
                    estimated_savings=Decimal(str(cost * 0.3)),
                    confidence_score=0.8,
                    description=f"Rightsize underutilized storage volume",
                    implementation_complexity="low",
                    risk_level="low"
                ))
        
        # Consolidation opportunities
        compute_resources = [r for r in resources if r.resource_type == ResourceType.COMPUTE]
        if len(compute_resources) >= 2:
            # Check if resources can be consolidated
            total_cpu = sum(r.performance_metrics.get('cpu_utilization', 0) for r in compute_resources)
            avg_cpu = total_cpu / len(compute_resources)
            
            if avg_cpu < 60:  # Moderate utilization threshold for consolidation
                total_cost = sum(r.cost_metrics.get('monthly_cost', 0) for r in compute_resources)
                if total_cost > 0:
                    recommendations.append(OptimizationOpportunity(
                        opportunity_id=f"consolidation_{hash(tuple(r.resource_id for r in compute_resources))}",
                        opportunity_type=OptimizationOpportunityType.CONSOLIDATION,
                        affected_resources=[r.resource_id for r in compute_resources],
                        estimated_savings=Decimal(str(total_cost * 0.3)),  # 30% savings from consolidation
                        confidence_score=0.7,
                        description=f"Consolidate {len(compute_resources)} compute resources",
                        implementation_complexity="high",
                        risk_level="medium",
                        prerequisites=["Workload compatibility analysis", "Migration planning"]
                    ))
        
        # Network optimization
        network_resources = [r for r in resources if r.resource_type == ResourceType.NETWORK]
        if len(network_resources) > 1:
            total_cost = sum(r.cost_metrics.get('monthly_cost', 0) for r in network_resources)
            if total_cost > 0:
                recommendations.append(OptimizationOpportunity(
                    opportunity_id=f"network_opt_{hash(tuple(r.resource_id for r in network_resources))}",
                    opportunity_type=OptimizationOpportunityType.NETWORK_OPTIMIZATION,
                    affected_resources=[r.resource_id for r in network_resources],
                    estimated_savings=Decimal(str(total_cost * 0.15)),
                    confidence_score=0.6,
                    description="Optimize network configuration and routing",
                    implementation_complexity="medium",
                    risk_level="low"
                ))
        
        # Load balancer optimization
        lb_resources = [r for r in resources if r.resource_type == ResourceType.LOAD_BALANCER]
        for lb in lb_resources:
            network_util = lb.performance_metrics.get('network_utilization', 0)
            cost = lb.cost_metrics.get('monthly_cost', 0)
            if network_util < 50 and cost > 0:  # Underutilized load balancer
                recommendations.append(OptimizationOpportunity(
                    opportunity_id=f"lb_opt_{lb.resource_id}",
                    opportunity_type=OptimizationOpportunityType.RIGHTSIZING,
                    affected_resources=[lb.resource_id],
                    estimated_savings=Decimal(str(cost * 0.2)),  # 20% savings
                    confidence_score=0.7,
                    description="Optimize load balancer configuration",
                    implementation_complexity="medium",
                    risk_level="low"
                ))
        
        return recommendations
    
    async def _determine_coordination_strategy(self, cluster: ResourceCluster, recommendations: List[OptimizationOpportunity]) -> str:
        """Determine the coordination strategy for cluster optimization"""
        
        # Count different types of optimizations
        consolidation_count = sum(1 for rec in recommendations if rec.opportunity_type == OptimizationOpportunityType.CONSOLIDATION)
        network_count = sum(1 for rec in recommendations if rec.opportunity_type == OptimizationOpportunityType.NETWORK_OPTIMIZATION)
        
        if consolidation_count > 0:
            return "sequential_with_validation"  # Consolidation requires careful sequencing
        elif network_count > 0:
            return "coordinated_parallel"  # Network changes can be done in parallel with coordination
        elif len(recommendations) > 3:
            return "phased_rollout"  # Many changes require phased approach
        else:
            return "parallel_execution"  # Simple changes can be done in parallel
    
    async def _plan_implementation_order(self, recommendations: List[OptimizationOpportunity], resources: List[ResourceNode]) -> List[str]:
        """Plan the order of implementation for recommendations"""
        
        # Sort by risk level and complexity
        risk_order = {"low": 0, "medium": 1, "high": 2}
        complexity_order = {"low": 0, "medium": 1, "high": 2}
        
        def sort_key(rec):
            risk_score = risk_order.get(rec.risk_level, 2)
            complexity_score = complexity_order.get(rec.implementation_complexity, 2)
            return (risk_score, complexity_score, -float(rec.estimated_savings))
        
        sorted_recommendations = sorted(recommendations, key=sort_key)
        
        return [rec.opportunity_id for rec in sorted_recommendations]


class GraphNeuralNetworkSystem:
    """Main system for graph neural network-based resource analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".GraphNeuralNetworkSystem")
        self.graph_builder = ResourceGraphBuilder()
        self.dependency_analyzer = DependencyAnalyzer()
        self.cascade_predictor = CascadePredictor()
        self.cluster_optimizer = ClusterOptimizer()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the graph neural network model"""
        try:
            # Calculate input dimension based on feature engineering
            input_dim = (
                len(ResourceType) +  # Resource type one-hot
                3 +  # Cost features
                3 +  # Performance features
                4    # Additional features
            )
            
            self.model = GraphNeuralNetwork(
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=64,
                num_layers=3
            )
            
            self.logger.info("Graph Neural Network model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing GNN model: {str(e)}")
            self.model = None
    
    async def build_resource_graph(self, account_id: str, resources: List[Dict[str, Any]] = None) -> ResourceGraph:
        """Build a resource graph for the given account"""
        try:
            if resources is None:
                # In a real implementation, this would fetch resources from cloud APIs
                resources = await self._fetch_resources_for_account(account_id)
            
            graph = await self.graph_builder.build_graph_from_resources(resources)
            
            # Add embeddings using the neural network if available
            if self.model is not None:
                graph = await self._add_neural_embeddings(graph)
            
            self.logger.info(f"Built resource graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error building resource graph: {str(e)}")
            raise FinOpsException(f"Failed to build resource graph: {str(e)}")
    
    async def analyze_dependencies(self, graph: ResourceGraph) -> DependencyAnalysis:
        """Analyze dependencies in the resource graph"""
        return await self.dependency_analyzer.analyze_dependencies(graph)
    
    async def predict_cascade_effects(self, action: Dict[str, Any], graph: ResourceGraph) -> CascadeEffects:
        """Predict cascade effects of an optimization action"""
        return await self.cascade_predictor.predict_cascade_effects(action, graph)
    
    async def optimize_resource_cluster(self, cluster: ResourceCluster, graph: ResourceGraph) -> ClusterOptimization:
        """Optimize a cluster of related resources"""
        return await self.cluster_optimizer.optimize_resource_cluster(cluster, graph)
    
    async def find_resource_clusters(self, graph: ResourceGraph) -> List[ResourceCluster]:
        """Find clusters of related resources in the graph"""
        try:
            # Convert to NetworkX for clustering
            nx_graph = self._to_networkx(graph)
            
            clusters = []
            
            # Use community detection for clustering
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(nx_graph.to_undirected())
                
                for i, community in enumerate(communities):
                    if len(community) >= 2:  # Only consider clusters with 2+ resources
                        cluster_resources = list(community)
                        
                        # Calculate cohesion score
                        subgraph = nx_graph.subgraph(cluster_resources)
                        edge_count = subgraph.number_of_edges()
                        # For directed graphs, max edges is n*(n-1) not n*(n-1)/2
                        max_edges = len(cluster_resources) * (len(cluster_resources) - 1)
                        cohesion_score = min(edge_count / max_edges if max_edges > 0 else 0, 1.0)
                        
                        # Calculate optimization potential
                        cluster_nodes = [node for node in graph.nodes if node.resource_id in cluster_resources]
                        avg_utilization = np.mean([
                            node.performance_metrics.get('cpu_utilization', 0) 
                            for node in cluster_nodes
                        ])
                        optimization_potential = max(0, (100 - avg_utilization) / 100)
                        
                        # Determine cluster type
                        resource_types = [node.resource_type for node in cluster_nodes]
                        if len(set(resource_types)) == 1:
                            cluster_type = f"homogeneous_{resource_types[0].value}"
                        else:
                            cluster_type = "heterogeneous"
                        
                        cluster = ResourceCluster(
                            cluster_id=f"cluster_{i}",
                            resources=cluster_resources,
                            cluster_type=cluster_type,
                            cohesion_score=cohesion_score,
                            optimization_potential=optimization_potential
                        )
                        clusters.append(cluster)
                
            except ImportError:
                # Fallback to simple connected components
                for component in nx.connected_components(nx_graph.to_undirected()):
                    if len(component) >= 2:
                        clusters.append(ResourceCluster(
                            cluster_id=f"component_{hash(tuple(component))}",
                            resources=list(component),
                            cluster_type="connected_component",
                            cohesion_score=0.5,
                            optimization_potential=0.3
                        ))
            
            self.logger.info(f"Found {len(clusters)} resource clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error finding resource clusters: {str(e)}")
            raise FinOpsException(f"Failed to find resource clusters: {str(e)}")
    
    def _to_networkx(self, graph: ResourceGraph) -> nx.DiGraph:
        """Convert ResourceGraph to NetworkX directed graph"""
        G = nx.DiGraph()
        
        for node in graph.nodes:
            G.add_node(node.resource_id, **node.to_dict())
        
        for edge in graph.edges:
            G.add_edge(edge.source_id, edge.target_id, weight=edge.strength, **edge.to_dict())
        
        return G
    
    async def _add_neural_embeddings(self, graph: ResourceGraph) -> ResourceGraph:
        """Add neural network embeddings to graph nodes"""
        try:
            if self.model is None:
                return graph
            
            # Convert to PyTorch Geometric format
            data = self.graph_builder.to_pytorch_geometric(graph)
            
            # Generate embeddings
            self.model.eval()
            with torch.no_grad():
                output = self.model(data.x, data.edge_index)
                embeddings = output['node_embeddings'].numpy()
            
            # Add embeddings to nodes
            for i, node in enumerate(graph.nodes):
                if i < len(embeddings):
                    node.embeddings = embeddings[i].tolist()
            
            return graph
            
        except Exception as e:
            self.logger.warning(f"Error adding neural embeddings: {str(e)}")
            return graph
    
    async def _fetch_resources_for_account(self, account_id: str) -> List[Dict[str, Any]]:
        """Fetch resources for an account (mock implementation)"""
        # This is a mock implementation - in reality, this would call cloud APIs
        mock_resources = [
            {
                'id': f'i-{account_id}001',
                'type': 'compute',
                'provider': 'aws',
                'region': 'us-east-1',
                'properties': {
                    'instance_type': 'm5.large',
                    'vpc_id': f'vpc-{account_id}001'
                },
                'cost_metrics': {
                    'monthly_cost': 100.0,
                    'hourly_cost': 0.14,
                    'utilization': 45.0
                },
                'performance_metrics': {
                    'cpu_utilization': 45.0,
                    'memory_utilization': 60.0,
                    'network_utilization': 20.0
                }
            },
            {
                'id': f'vol-{account_id}001',
                'type': 'storage',
                'provider': 'aws',
                'region': 'us-east-1',
                'properties': {
                    'volume_type': 'gp3',
                    'size_gb': 100
                },
                'cost_metrics': {
                    'monthly_cost': 10.0,
                    'utilization': 30.0
                },
                'performance_metrics': {
                    'storage_utilization': 30.0
                }
            },
            {
                'id': f'lb-{account_id}001',
                'type': 'load_balancer',
                'provider': 'aws',
                'region': 'us-east-1',
                'properties': {
                    'target_instances': [f'i-{account_id}001']
                },
                'cost_metrics': {
                    'monthly_cost': 25.0
                },
                'performance_metrics': {
                    'network_utilization': 15.0
                }
            }
        ]
        
        return mock_resources