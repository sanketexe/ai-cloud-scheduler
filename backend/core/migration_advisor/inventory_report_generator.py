"""
Inventory Report Generator for Cloud Migration Advisor

This module provides customizable report generation with grouping, aggregation,
and export functionality.

Requirements: 6.5
"""

import logging
import json
import csv
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from collections import defaultdict

from .resource_discovery_engine import CloudResource, ResourceType
from .auto_categorization_engine import CategorizedResources, ResourceCategorization
from .organizational_structure_manager import DimensionType, OrganizationalStructure


logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


class GroupingLevel(Enum):
    """Grouping levels for reports"""
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"


class AggregationFunction(Enum):
    """Aggregation functions for reports"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    LIST = "list"


@dataclass
class ReportColumn:
    """Definition of a report column"""
    name: str
    field: str
    display_name: Optional[str] = None
    formatter: Optional[Callable[[Any], str]] = None
    aggregation: Optional[AggregationFunction] = None
    
    def get_display_name(self) -> str:
        """Get display name for column"""
        return self.display_name or self.name.replace('_', ' ').title()


@dataclass
class ReportGrouping:
    """
    Grouping configuration for reports
    
    Requirements: 6.5
    """
    dimensions: List[DimensionType] = field(default_factory=list)
    level: GroupingLevel = GroupingLevel.SINGLE
    include_totals: bool = True
    include_subtotals: bool = True


@dataclass
class AggregationRule:
    """Rule for aggregating data in reports"""
    field: str
    function: AggregationFunction
    label: Optional[str] = None
    
    def get_label(self) -> str:
        """Get label for aggregation"""
        return self.label or f"{self.function.value}_{self.field}"


@dataclass
class ReportOptions:
    """Options for report generation"""
    title: Optional[str] = None
    description: Optional[str] = None
    include_metadata: bool = True
    include_summary: bool = True
    max_rows: Optional[int] = None
    sort_by: Optional[str] = None
    sort_descending: bool = False
    filters: Optional[Dict[str, Any]] = None


@dataclass
class ReportSection:
    """A section in a report"""
    title: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    subsections: List['ReportSection'] = field(default_factory=list)


@dataclass
class InventoryReport:
    """
    Complete inventory report
    
    Requirements: 6.5
    """
    report_id: str
    title: str
    generated_at: datetime
    format: ReportFormat
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    
    def get_total_resources(self) -> int:
        """Get total number of resources in report"""
        return self.summary.get('total_resources', 0)


class InventoryReportGenerator:
    """
    Generator for customizable inventory reports
    
    Requirements: 6.5
    """
    
    def __init__(self):
        """Initialize the inventory report generator"""
        logger.info("Inventory Report Generator initialized")
    
    def generate_report(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        grouping: Optional[ReportGrouping] = None,
        aggregations: Optional[List[AggregationRule]] = None,
        columns: Optional[List[ReportColumn]] = None,
        options: Optional[ReportOptions] = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> InventoryReport:
        """
        Generate an inventory report
        
        Args:
            resources: List of cloud resources
            categorizations: Resource categorizations
            structure: Organizational structure
            grouping: Optional grouping configuration
            aggregations: Optional aggregation rules
            columns: Optional column definitions
            options: Optional report options
            format: Report format
            
        Returns:
            InventoryReport
        """
        logger.info(f"Generating inventory report for {len(resources)} resources")
        
        if options is None:
            options = ReportOptions()
        
        if columns is None:
            columns = self._get_default_columns()
        
        # Generate report ID
        report_id = self._generate_report_id()
        
        # Create report
        report = InventoryReport(
            report_id=report_id,
            title=options.title or "Resource Inventory Report",
            generated_at=datetime.utcnow(),
            format=format
        )
        
        # Apply filters if specified
        if options.filters:
            resources = self._apply_filters(resources, categorizations, options.filters)
        
        # Apply sorting if specified
        if options.sort_by:
            resources = self._sort_resources(
                resources,
                categorizations,
                options.sort_by,
                options.sort_descending
            )
        
        # Apply max rows limit
        if options.max_rows:
            resources = resources[:options.max_rows]
        
        # Generate sections based on grouping
        if grouping and grouping.dimensions:
            report.sections = self._generate_grouped_sections(
                resources,
                categorizations,
                structure,
                grouping,
                aggregations,
                columns
            )
        else:
            # Single section with all resources
            section = self._generate_section(
                "All Resources",
                resources,
                categorizations,
                aggregations,
                columns
            )
            report.sections = [section]
        
        # Generate summary
        if options.include_summary:
            report.summary = self._generate_summary(
                resources,
                categorizations,
                aggregations
            )
        
        # Add metadata
        if options.include_metadata:
            report.metadata = self._generate_metadata(
                resources,
                categorizations,
                structure,
                options
            )
        
        # Export to specified format
        report.content = self._export_report(report, format)
        
        logger.info(f"Report generated: {report_id}")
        
        return report
    
    def generate_team_report(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        format: ReportFormat = ReportFormat.JSON
    ) -> InventoryReport:
        """Generate report grouped by team"""
        grouping = ReportGrouping(
            dimensions=[DimensionType.TEAM],
            level=GroupingLevel.SINGLE
        )
        
        options = ReportOptions(
            title="Resources by Team"
        )
        
        return self.generate_report(
            resources,
            categorizations,
            structure,
            grouping=grouping,
            options=options,
            format=format
        )
    
    def generate_project_report(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        format: ReportFormat = ReportFormat.JSON
    ) -> InventoryReport:
        """Generate report grouped by project"""
        grouping = ReportGrouping(
            dimensions=[DimensionType.PROJECT],
            level=GroupingLevel.SINGLE
        )
        
        options = ReportOptions(
            title="Resources by Project"
        )
        
        return self.generate_report(
            resources,
            categorizations,
            structure,
            grouping=grouping,
            options=options,
            format=format
        )
    
    def generate_environment_report(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        format: ReportFormat = ReportFormat.JSON
    ) -> InventoryReport:
        """Generate report grouped by environment"""
        grouping = ReportGrouping(
            dimensions=[DimensionType.ENVIRONMENT],
            level=GroupingLevel.SINGLE
        )
        
        options = ReportOptions(
            title="Resources by Environment"
        )
        
        return self.generate_report(
            resources,
            categorizations,
            structure,
            grouping=grouping,
            options=options,
            format=format
        )
    
    def generate_multi_dimensional_report(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        dimensions: List[DimensionType],
        format: ReportFormat = ReportFormat.JSON
    ) -> InventoryReport:
        """Generate report with multi-level grouping"""
        grouping = ReportGrouping(
            dimensions=dimensions,
            level=GroupingLevel.MULTI
        )
        
        options = ReportOptions(
            title=f"Resources by {', '.join(d.value for d in dimensions)}"
        )
        
        return self.generate_report(
            resources,
            categorizations,
            structure,
            grouping=grouping,
            options=options,
            format=format
        )
    
    def export_to_csv(self, report: InventoryReport) -> str:
        """Export report to CSV format"""
        return self._export_to_csv(report)
    
    def export_to_json(self, report: InventoryReport) -> str:
        """Export report to JSON format"""
        return self._export_to_json(report)
    
    def export_to_html(self, report: InventoryReport) -> str:
        """Export report to HTML format"""
        return self._export_to_html(report)
    
    def export_to_markdown(self, report: InventoryReport) -> str:
        """Export report to Markdown format"""
        return self._export_to_markdown(report)
    
    # Private helper methods
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        import hashlib
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _get_default_columns(self) -> List[ReportColumn]:
        """Get default column definitions"""
        return [
            ReportColumn(name="resource_id", field="resource_id", display_name="Resource ID"),
            ReportColumn(name="resource_name", field="resource_name", display_name="Name"),
            ReportColumn(name="resource_type", field="resource_type", display_name="Type"),
            ReportColumn(name="provider", field="provider", display_name="Provider"),
            ReportColumn(name="region", field="region", display_name="Region"),
            ReportColumn(name="team", field="team", display_name="Team"),
            ReportColumn(name="project", field="project", display_name="Project"),
            ReportColumn(name="environment", field="environment", display_name="Environment")
        ]
    
    def _apply_filters(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        filters: Dict[str, Any]
    ) -> List[CloudResource]:
        """Apply filters to resources"""
        filtered = []
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            match = True
            for field, value in filters.items():
                resource_value = self._get_resource_field(resource, categorization, field)
                if resource_value != value:
                    match = False
                    break
            
            if match:
                filtered.append(resource)
        
        return filtered
    
    def _sort_resources(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        sort_by: str,
        descending: bool
    ) -> List[CloudResource]:
        """Sort resources by field"""
        def sort_key(resource: CloudResource) -> Any:
            categorization = categorizations.get_categorization(resource.resource_id)
            return self._get_resource_field(resource, categorization, sort_by)
        
        return sorted(resources, key=sort_key, reverse=descending)
    
    def _generate_grouped_sections(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        grouping: ReportGrouping,
        aggregations: Optional[List[AggregationRule]],
        columns: List[ReportColumn]
    ) -> List[ReportSection]:
        """Generate sections with grouping"""
        sections = []
        
        # Group resources by first dimension
        primary_dimension = grouping.dimensions[0]
        grouped = self._group_resources(resources, categorizations, primary_dimension)
        
        for group_value, group_resources in grouped.items():
            section = self._generate_section(
                f"{primary_dimension.value.title()}: {group_value}",
                group_resources,
                categorizations,
                aggregations,
                columns
            )
            
            # If multi-level grouping, create subsections
            if grouping.level == GroupingLevel.MULTI and len(grouping.dimensions) > 1:
                section.subsections = self._generate_subsections(
                    group_resources,
                    categorizations,
                    grouping.dimensions[1:],
                    aggregations,
                    columns
                )
            
            sections.append(section)
        
        return sections
    
    def _generate_subsections(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        dimensions: List[DimensionType],
        aggregations: Optional[List[AggregationRule]],
        columns: List[ReportColumn]
    ) -> List[ReportSection]:
        """Generate subsections recursively"""
        if not dimensions:
            return []
        
        subsections = []
        dimension = dimensions[0]
        grouped = self._group_resources(resources, categorizations, dimension)
        
        for group_value, group_resources in grouped.items():
            subsection = self._generate_section(
                f"{dimension.value.title()}: {group_value}",
                group_resources,
                categorizations,
                aggregations,
                columns
            )
            
            # Recurse for remaining dimensions
            if len(dimensions) > 1:
                subsection.subsections = self._generate_subsections(
                    group_resources,
                    categorizations,
                    dimensions[1:],
                    aggregations,
                    columns
                )
            
            subsections.append(subsection)
        
        return subsections
    
    def _generate_section(
        self,
        title: str,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        aggregations: Optional[List[AggregationRule]],
        columns: List[ReportColumn]
    ) -> ReportSection:
        """Generate a report section"""
        section = ReportSection(title=title)
        
        # Generate data rows
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            row = self._generate_row(resource, categorization, columns)
            section.data.append(row)
        
        # Calculate aggregations
        if aggregations:
            section.aggregations = self._calculate_aggregations(
                resources,
                categorizations,
                aggregations
            )
        
        return section
    
    def _generate_row(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization],
        columns: List[ReportColumn]
    ) -> Dict[str, Any]:
        """Generate a data row for a resource"""
        row = {}
        
        for column in columns:
            value = self._get_resource_field(resource, categorization, column.field)
            
            # Apply formatter if specified
            if column.formatter:
                value = column.formatter(value)
            
            row[column.name] = value
        
        return row
    
    def _get_resource_field(
        self,
        resource: CloudResource,
        categorization: Optional[ResourceCategorization],
        field: str
    ) -> Any:
        """Get field value from resource or categorization"""
        # Resource fields
        if field == "resource_id":
            return resource.resource_id
        elif field == "resource_name":
            return resource.resource_name
        elif field == "resource_type":
            return resource.resource_type.value if hasattr(resource.resource_type, 'value') else resource.resource_type
        elif field == "provider":
            return resource.provider.value if hasattr(resource.provider, 'value') else resource.provider
        elif field == "region":
            return resource.region
        elif field == "created_date":
            return resource.created_date
        
        # Categorization fields
        if categorization:
            if field == "team":
                return categorization.team
            elif field == "project":
                return categorization.project
            elif field == "environment":
                return categorization.environment
            elif field == "cost_center":
                return categorization.cost_center
        
        return None
    
    def _group_resources(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        dimension: DimensionType
    ) -> Dict[str, List[CloudResource]]:
        """Group resources by dimension"""
        grouped: Dict[str, List[CloudResource]] = defaultdict(list)
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            
            # Get dimension value
            if dimension == DimensionType.TEAM:
                value = categorization.team if categorization else None
            elif dimension == DimensionType.PROJECT:
                value = categorization.project if categorization else None
            elif dimension == DimensionType.ENVIRONMENT:
                value = categorization.environment if categorization else None
            elif dimension == DimensionType.REGION:
                value = resource.region
            elif dimension == DimensionType.COST_CENTER:
                value = categorization.cost_center if categorization else None
            else:
                value = None
            
            group_key = value or "Uncategorized"
            grouped[group_key].append(resource)
        
        return dict(grouped)
    
    def _calculate_aggregations(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        aggregations: List[AggregationRule]
    ) -> Dict[str, Any]:
        """Calculate aggregations for resources"""
        results = {}
        
        for agg in aggregations:
            if agg.function == AggregationFunction.COUNT:
                results[agg.get_label()] = len(resources)
            elif agg.function == AggregationFunction.LIST:
                values = [
                    self._get_resource_field(r, categorizations.get_categorization(r.resource_id), agg.field)
                    for r in resources
                ]
                results[agg.get_label()] = list(set(values))
        
        return results
    
    def _generate_summary(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        aggregations: Optional[List[AggregationRule]]
    ) -> Dict[str, Any]:
        """Generate report summary"""
        summary = {
            "total_resources": len(resources),
            "by_type": self._count_by_field(resources, categorizations, "resource_type"),
            "by_provider": self._count_by_field(resources, categorizations, "provider"),
            "by_region": self._count_by_field(resources, categorizations, "region")
        }
        
        return summary
    
    def _count_by_field(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        field: str
    ) -> Dict[str, int]:
        """Count resources by field value"""
        counts: Dict[str, int] = defaultdict(int)
        
        for resource in resources:
            categorization = categorizations.get_categorization(resource.resource_id)
            value = self._get_resource_field(resource, categorization, field)
            counts[str(value)] += 1
        
        return dict(counts)
    
    def _generate_metadata(
        self,
        resources: List[CloudResource],
        categorizations: CategorizedResources,
        structure: OrganizationalStructure,
        options: ReportOptions
    ) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_resources": len(resources),
            "structure_name": structure.name,
            "options": {
                "title": options.title,
                "description": options.description,
                "max_rows": options.max_rows,
                "sort_by": options.sort_by
            }
        }
    
    def _export_report(self, report: InventoryReport, format: ReportFormat) -> str:
        """Export report to specified format"""
        if format == ReportFormat.JSON:
            return self._export_to_json(report)
        elif format == ReportFormat.CSV:
            return self._export_to_csv(report)
        elif format == ReportFormat.HTML:
            return self._export_to_html(report)
        elif format == ReportFormat.MARKDOWN:
            return self._export_to_markdown(report)
        
        return ""
    
    def _export_to_json(self, report: InventoryReport) -> str:
        """Export to JSON"""
        data = {
            "report_id": report.report_id,
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary,
            "metadata": report.metadata,
            "sections": [
                {
                    "title": section.title,
                    "data": section.data,
                    "aggregations": section.aggregations
                }
                for section in report.sections
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def _export_to_csv(self, report: InventoryReport) -> str:
        """Export to CSV"""
        output = StringIO()
        
        # Get all data rows from all sections
        all_rows = []
        for section in report.sections:
            all_rows.extend(section.data)
        
        if not all_rows:
            return ""
        
        # Write CSV
        fieldnames = list(all_rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
        return output.getvalue()
    
    def _export_to_html(self, report: InventoryReport) -> str:
        """Export to HTML"""
        html = f"<html><head><title>{report.title}</title></head><body>"
        html += f"<h1>{report.title}</h1>"
        html += f"<p>Generated: {report.generated_at.isoformat()}</p>"
        
        for section in report.sections:
            html += f"<h2>{section.title}</h2>"
            html += "<table border='1'>"
            
            if section.data:
                # Header
                html += "<tr>"
                for key in section.data[0].keys():
                    html += f"<th>{key}</th>"
                html += "</tr>"
                
                # Rows
                for row in section.data:
                    html += "<tr>"
                    for value in row.values():
                        html += f"<td>{value}</td>"
                    html += "</tr>"
            
            html += "</table>"
        
        html += "</body></html>"
        return html
    
    def _export_to_markdown(self, report: InventoryReport) -> str:
        """Export to Markdown"""
        md = f"# {report.title}\n\n"
        md += f"Generated: {report.generated_at.isoformat()}\n\n"
        
        # Summary
        if report.summary:
            md += "## Summary\n\n"
            md += f"Total Resources: {report.summary.get('total_resources', 0)}\n\n"
        
        # Sections
        for section in report.sections:
            md += f"## {section.title}\n\n"
            
            if section.data:
                # Table header
                keys = list(section.data[0].keys())
                md += "| " + " | ".join(keys) + " |\n"
                md += "| " + " | ".join(["---"] * len(keys)) + " |\n"
                
                # Table rows
                for row in section.data:
                    md += "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |\n"
                
                md += "\n"
        
        return md
