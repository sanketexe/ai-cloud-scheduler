"""
FastAPI endpoints for Graph Neural Network System
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging

from .graph_neural_network_system import (
    GraphNeuralNetworkSystem, ResourceGraph, DependencyAnalysis,
    CascadeEffects, ResourceCluster, ClusterOptimization,
    OptimizationOpportunity, ResourceNode, ResourceEdge
)
from .auth import get_current_user
from .models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph-analysis", tags=["Graph Neural Network"])

# Initialize the system
gnn_system = GraphNeuralNetworkSystem()


# Pydantic models for API
class ResourceGraphResponse(BaseModel):
    """Response model for resource graph"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DependencyAnalysisResponse(BaseModel):
    """Response model for dependency analysis"""
    critical_paths: List[List[str]]
    bottlenecks: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    dependency_matrix: Dict[str, List[str]]


class CascadeEffectsResponse(BaseModel):
    """Response model for cascade effects prediction"""
    primary_impact: Dict[str, Any]
    secondary_impacts: List[Dict[str, Any]]
    risk_score: float
    affected_resources: List[str]
    mitigation_strategies: List[str]


class ResourceClusterResponse(BaseModel):
    """Response model for resource cluster"""
    cluster_id: str
    resources: List[str]
    cluster_type: str
    cohesion_score: float
    optimization_potential: float


class ClusterOptimizationResponse(BaseModel):
    """Response model for cluster optimization"""
    cluster_id: str
    recommendations: List[Dict[str, Any]]
    coordination_strategy: str
    implementation_order: List[str]
    total_estimated_savings: float


class OptimizationActionRequest(BaseModel):
    """Request model for optimization action"""
    target_resource_id: str
    action_type: str = Field(..., description="Type of optimization action (rightsizing, decommission, migration)")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ResourceInput(BaseModel):
    """Input model for resource data"""
    id: str
    type: str
    provider: str
    region: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    cost_metrics: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


@router.post("/build-graph/{account_id}", response_model=ResourceGraphResponse)
async def build_resource_graph(
    account_id: str,
    resources: Optional[List[ResourceInput]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Build a resource graph for the specified account
    """
    try:
        logger.info(f"Building resource graph for account {account_id}")
        
        # Convert Pydantic models to dictionaries if provided
        resource_data = None
        if resources:
            resource_data = [resource.dict() for resource in resources]
        
        # Build the graph
        graph = await gnn_system.build_resource_graph(account_id, resource_data)
        
        return ResourceGraphResponse(
            nodes=[node.to_dict() for node in graph.nodes],
            edges=[edge.to_dict() for edge in graph.edges],
            metadata=graph.metadata
        )
        
    except Exception as e:
        logger.error(f"Error building resource graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to build resource graph: {str(e)}")


@router.post("/analyze-dependencies", response_model=DependencyAnalysisResponse)
async def analyze_dependencies(
    graph_data: ResourceGraphResponse,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze dependencies in a resource graph
    """
    try:
        logger.info("Analyzing resource dependencies")
        
        # Convert response model back to ResourceGraph
        graph = _convert_to_resource_graph(graph_data)
        
        # Perform dependency analysis
        analysis = await gnn_system.analyze_dependencies(graph)
        
        return DependencyAnalysisResponse(
            critical_paths=analysis.critical_paths,
            bottlenecks=analysis.bottlenecks,
            optimization_opportunities=[opp.to_dict() for opp in analysis.optimization_opportunities],
            risk_assessment=analysis.risk_assessment,
            dependency_matrix=analysis.dependency_matrix
        )
        
    except Exception as e:
        logger.error(f"Error analyzing dependencies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze dependencies: {str(e)}")


@router.post("/predict-cascade-effects", response_model=CascadeEffectsResponse)
async def predict_cascade_effects(
    action: OptimizationActionRequest,
    graph_data: ResourceGraphResponse,
    current_user: User = Depends(get_current_user)
):
    """
    Predict cascade effects of an optimization action
    """
    try:
        logger.info(f"Predicting cascade effects for action on {action.target_resource_id}")
        
        # Convert response model back to ResourceGraph
        graph = _convert_to_resource_graph(graph_data)
        
        # Prepare action data
        action_data = {
            'target_resource_id': action.target_resource_id,
            'action_type': action.action_type,
            **action.parameters
        }
        
        # Predict cascade effects
        effects = await gnn_system.predict_cascade_effects(action_data, graph)
        
        return CascadeEffectsResponse(
            primary_impact=effects.primary_impact,
            secondary_impacts=effects.secondary_impacts,
            risk_score=effects.risk_score,
            affected_resources=effects.affected_resources,
            mitigation_strategies=effects.mitigation_strategies
        )
        
    except Exception as e:
        logger.error(f"Error predicting cascade effects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to predict cascade effects: {str(e)}")


@router.post("/find-clusters", response_model=List[ResourceClusterResponse])
async def find_resource_clusters(
    graph_data: ResourceGraphResponse,
    min_cluster_size: int = Query(2, description="Minimum cluster size"),
    current_user: User = Depends(get_current_user)
):
    """
    Find clusters of related resources in the graph
    """
    try:
        logger.info("Finding resource clusters")
        
        # Convert response model back to ResourceGraph
        graph = _convert_to_resource_graph(graph_data)
        
        # Find clusters
        clusters = await gnn_system.find_resource_clusters(graph)
        
        # Filter by minimum size
        filtered_clusters = [cluster for cluster in clusters if len(cluster.resources) >= min_cluster_size]
        
        return [
            ResourceClusterResponse(
                cluster_id=cluster.cluster_id,
                resources=cluster.resources,
                cluster_type=cluster.cluster_type,
                cohesion_score=cluster.cohesion_score,
                optimization_potential=cluster.optimization_potential
            )
            for cluster in filtered_clusters
        ]
        
    except Exception as e:
        logger.error(f"Error finding resource clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find resource clusters: {str(e)}")


@router.post("/optimize-cluster", response_model=ClusterOptimizationResponse)
async def optimize_resource_cluster(
    cluster: ResourceClusterResponse,
    graph_data: ResourceGraphResponse,
    current_user: User = Depends(get_current_user)
):
    """
    Optimize a cluster of related resources
    """
    try:
        logger.info(f"Optimizing resource cluster {cluster.cluster_id}")
        
        # Convert response models back to domain objects
        graph = _convert_to_resource_graph(graph_data)
        cluster_obj = ResourceCluster(
            cluster_id=cluster.cluster_id,
            resources=cluster.resources,
            cluster_type=cluster.cluster_type,
            cohesion_score=cluster.cohesion_score,
            optimization_potential=cluster.optimization_potential
        )
        
        # Optimize cluster
        optimization = await gnn_system.optimize_resource_cluster(cluster_obj, graph)
        
        return ClusterOptimizationResponse(
            cluster_id=optimization.cluster_id,
            recommendations=[rec.to_dict() for rec in optimization.recommendations],
            coordination_strategy=optimization.coordination_strategy,
            implementation_order=optimization.implementation_order,
            total_estimated_savings=float(optimization.total_estimated_savings)
        )
        
    except Exception as e:
        logger.error(f"Error optimizing resource cluster: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize resource cluster: {str(e)}")


@router.get("/graph-analysis/{account_id}/summary")
async def get_graph_analysis_summary(
    account_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a comprehensive summary of graph analysis for an account
    """
    try:
        logger.info(f"Getting graph analysis summary for account {account_id}")
        
        # Build the graph
        graph = await gnn_system.build_resource_graph(account_id)
        
        # Perform analysis
        dependency_analysis = await gnn_system.analyze_dependencies(graph)
        clusters = await gnn_system.find_resource_clusters(graph)
        
        # Calculate summary statistics
        total_resources = len(graph.nodes)
        total_relationships = len(graph.edges)
        total_clusters = len(clusters)
        total_opportunities = len(dependency_analysis.optimization_opportunities)
        total_potential_savings = sum(
            float(opp.estimated_savings) 
            for opp in dependency_analysis.optimization_opportunities
        )
        
        # Risk assessment
        risk_score = dependency_analysis.risk_assessment.get('overall_risk_score', 0.0)
        bottleneck_count = len(dependency_analysis.bottlenecks)
        critical_path_count = len(dependency_analysis.critical_paths)
        
        return {
            'account_id': account_id,
            'summary': {
                'total_resources': total_resources,
                'total_relationships': total_relationships,
                'total_clusters': total_clusters,
                'total_optimization_opportunities': total_opportunities,
                'total_potential_savings': total_potential_savings,
                'overall_risk_score': risk_score,
                'bottleneck_count': bottleneck_count,
                'critical_path_count': critical_path_count
            },
            'top_opportunities': [
                opp.to_dict() for opp in 
                sorted(dependency_analysis.optimization_opportunities, 
                       key=lambda x: float(x.estimated_savings), reverse=True)[:5]
            ],
            'high_risk_resources': dependency_analysis.bottlenecks[:5],
            'largest_clusters': [
                {
                    'cluster_id': cluster.cluster_id,
                    'resource_count': len(cluster.resources),
                    'cluster_type': cluster.cluster_type,
                    'optimization_potential': cluster.optimization_potential
                }
                for cluster in sorted(clusters, key=lambda x: len(x.resources), reverse=True)[:5]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting graph analysis summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph analysis summary: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the Graph Neural Network system
    """
    try:
        # Check if the system is properly initialized
        system_status = {
            'status': 'healthy',
            'gnn_model_loaded': gnn_system.model is not None,
            'components': {
                'graph_builder': 'operational',
                'dependency_analyzer': 'operational',
                'cascade_predictor': 'operational',
                'cluster_optimizer': 'operational'
            }
        }
        
        return system_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def _convert_to_resource_graph(graph_data: ResourceGraphResponse) -> ResourceGraph:
    """Convert API response model back to ResourceGraph domain object"""
    from .graph_neural_network_system import ResourceType, RelationshipType
    
    # Convert nodes
    nodes = []
    for node_data in graph_data.nodes:
        node = ResourceNode(
            resource_id=node_data['resource_id'],
            resource_type=ResourceType(node_data['resource_type']),
            provider=node_data['provider'],
            region=node_data['region'],
            properties=node_data.get('properties', {}),
            cost_metrics=node_data.get('cost_metrics', {}),
            performance_metrics=node_data.get('performance_metrics', {}),
            embeddings=node_data.get('embeddings')
        )
        nodes.append(node)
    
    # Convert edges
    edges = []
    for edge_data in graph_data.edges:
        edge = ResourceEdge(
            source_id=edge_data['source_id'],
            target_id=edge_data['target_id'],
            relationship_type=RelationshipType(edge_data['relationship_type']),
            strength=edge_data['strength'],
            properties=edge_data.get('properties', {})
        )
        edges.append(edge)
    
    return ResourceGraph(
        nodes=nodes,
        edges=edges,
        metadata=graph_data.metadata
    )