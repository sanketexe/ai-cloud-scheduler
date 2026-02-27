"""
PDF export service for migration recommendation reports.

This module provides functionality to export comprehensive migration reports
to PDF format for sharing and documentation purposes.
"""

import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from decimal import Decimal

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .report_generator import ComprehensiveReport


class PDFExportService:
    """Service for exporting migration reports to PDF format"""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF export. "
                "Install with: pip install reportlab"
            )
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1976d2'),
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1976d2'),
            borderWidth=1,
            borderColor=colors.HexColor('#1976d2'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#424242')
        ))
        
        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            backColor=colors.HexColor('#f5f5f5'),
            borderWidth=1,
            borderColor=colors.HexColor('#e0e0e0'),
            borderPadding=10
        ))
        
        # Recommendation box style
        self.styles.add(ParagraphStyle(
            name='RecommendationBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            backColor=colors.HexColor('#e8f5e8'),
            borderWidth=2,
            borderColor=colors.HexColor('#4caf50'),
            borderPadding=15
        ))
    
    def export_to_pdf(self, report: ComprehensiveReport) -> bytes:
        """
        Export a comprehensive report to PDF format.
        
        Args:
            report: The comprehensive report to export
            
        Returns:
            bytes: PDF file content as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build document content
        story = []
        
        # Title page
        story.extend(self._build_title_page(report))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._build_table_of_contents())
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._build_executive_summary(report))
        story.append(PageBreak())
        
        # Technical analysis
        story.extend(self._build_technical_analysis(report))
        story.append(PageBreak())
        
        # Implementation roadmap
        story.extend(self._build_implementation_roadmap(report))
        story.append(PageBreak())
        
        # Assessment inputs
        story.extend(self._build_assessment_inputs(report))
        
        # Appendices
        if report.appendices:
            story.append(PageBreak())
            story.extend(self._build_appendices(report))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _build_title_page(self, report: ComprehensiveReport) -> List[Any]:
        """Build the title page"""
        story = []
        
        # Main title
        story.append(Paragraph(
            "Cloud Migration Recommendation Report",
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Organization name
        story.append(Paragraph(
            f"<b>{report.executive_summary.organization_name}</b>",
            self.styles['Title']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendation highlight
        recommendation_text = f"""
        <b>Primary Recommendation: {report.executive_summary.primary_recommendation}</b><br/>
        Confidence Score: {report.executive_summary.confidence_score:.1%}<br/>
        Generated: {report.generated_at.strftime('%B %d, %Y')}
        """
        story.append(Paragraph(recommendation_text, self.styles['RecommendationBox']))
        story.append(Spacer(1, 0.5*inch))
        
        # Key metrics table
        if report.executive_summary.estimated_monthly_cost:
            metrics_data = [
                ['Metric', 'Value'],
                ['Estimated Monthly Cost', f"${report.executive_summary.estimated_monthly_cost:,.2f}"],
                ['Migration Duration', f"{report.executive_summary.migration_duration_weeks} weeks"],
            ]
            
            if report.executive_summary.estimated_savings:
                metrics_data.append(['Estimated Monthly Savings', f"${report.executive_summary.estimated_savings:,.2f}"])
            
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
        
        story.append(Spacer(1, 1*inch))
        
        # Footer
        story.append(Paragraph(
            f"Report ID: {report.report_id}",
            self.styles['Normal']
        ))
        
        return story
    
    def _build_table_of_contents(self) -> List[Any]:
        """Build table of contents"""
        story = []
        
        story.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        toc_data = [
            ['Section', 'Page'],
            ['Executive Summary', '3'],
            ['Technical Analysis', '4'],
            ['Implementation Roadmap', '5'],
            ['Assessment Inputs', '6'],
            ['Appendices', '7'],
        ]
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(toc_table)
        
        return story
    
    def _build_executive_summary(self, report: ComprehensiveReport) -> List[Any]:
        """Build executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Summary paragraph
        summary_text = f"""
        Based on our comprehensive assessment of {report.executive_summary.organization_name}'s 
        infrastructure and requirements, we recommend <b>{report.executive_summary.primary_recommendation}</b> 
        as the optimal cloud provider for your migration. This recommendation is based on a detailed 
        analysis with a confidence score of {report.executive_summary.confidence_score:.1%}.
        """
        story.append(Paragraph(summary_text, self.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key benefits
        if report.executive_summary.key_benefits:
            story.append(Paragraph("Key Benefits", self.styles['SubsectionHeader']))
            for benefit in report.executive_summary.key_benefits:
                story.append(Paragraph(f"• {benefit}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Critical considerations
        if report.executive_summary.critical_considerations:
            story.append(Paragraph("Critical Considerations", self.styles['SubsectionHeader']))
            for consideration in report.executive_summary.critical_considerations:
                story.append(Paragraph(f"• {consideration}", self.styles['Normal']))
        
        return story
    
    def _build_technical_analysis(self, report: ComprehensiveReport) -> List[Any]:
        """Build technical analysis section"""
        story = []
        
        story.append(Paragraph("Technical Analysis", self.styles['SectionHeader']))
        
        # Workload summary
        story.append(Paragraph("Workload Summary", self.styles['SubsectionHeader']))
        workload = report.technical_analysis.workload_summary
        
        workload_data = [
            ['Resource', 'Current Requirement'],
            ['Compute Cores', str(workload.get('total_compute_cores', 'N/A'))],
            ['Memory (GB)', str(workload.get('total_memory_gb', 'N/A'))],
            ['Storage (TB)', str(workload.get('total_storage_tb', 'N/A'))],
            ['Applications', str(workload.get('application_count', 'N/A'))],
        ]
        
        workload_table = Table(workload_data, colWidths=[2*inch, 2*inch])
        workload_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3f2fd')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(workload_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Provider comparison
        story.append(Paragraph("Provider Evaluation Results", self.styles['SubsectionHeader']))
        
        if report.technical_analysis.provider_evaluations:
            eval_data = [['Provider', 'Overall Score', 'Monthly Cost', 'Migration Time']]
            
            for evaluation in report.technical_analysis.provider_evaluations:
                cost_str = f"${evaluation['estimated_monthly_cost']:,.2f}" if evaluation['estimated_monthly_cost'] else 'TBD'
                eval_data.append([
                    evaluation['provider'],
                    f"{evaluation['overall_score']:.1f}",
                    cost_str,
                    f"{evaluation['migration_duration_weeks']} weeks"
                ])
            
            eval_table = Table(eval_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch])
            eval_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e8')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(eval_table)
        
        return story
    
    def _build_implementation_roadmap(self, report: ComprehensiveReport) -> List[Any]:
        """Build implementation roadmap section"""
        story = []
        
        story.append(Paragraph("Implementation Roadmap", self.styles['SectionHeader']))
        
        # Timeline overview
        timeline = report.implementation_roadmap.timeline_overview
        story.append(Paragraph("Timeline Overview", self.styles['SubsectionHeader']))
        story.append(Paragraph(
            f"Total Duration: {timeline['total_duration_weeks']} weeks<br/>"
            f"Estimated Start: {timeline['estimated_start_date']}<br/>"
            f"Estimated Completion: {timeline['estimated_completion_date']}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Migration phases
        story.append(Paragraph("Migration Phases", self.styles['SubsectionHeader']))
        
        for phase in report.implementation_roadmap.migration_phases:
            story.append(Paragraph(
                f"<b>{phase['phase']}</b> ({phase['duration_weeks']} weeks)",
                self.styles['Normal']
            ))
            story.append(Paragraph(phase['description'], self.styles['Normal']))
            
            if phase.get('key_activities'):
                for activity in phase['key_activities']:
                    story.append(Paragraph(f"  • {activity}", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Success criteria
        if report.implementation_roadmap.success_criteria:
            story.append(Paragraph("Success Criteria", self.styles['SubsectionHeader']))
            for criterion in report.implementation_roadmap.success_criteria:
                story.append(Paragraph(f"• {criterion}", self.styles['Normal']))
        
        return story
    
    def _build_assessment_inputs(self, report: ComprehensiveReport) -> List[Any]:
        """Build assessment inputs section"""
        story = []
        
        story.append(Paragraph("Assessment Inputs", self.styles['SectionHeader']))
        
        # Organization profile
        story.append(Paragraph("Organization Profile", self.styles['SubsectionHeader']))
        org = report.assessment_inputs.organization_profile
        
        org_data = [
            ['Attribute', 'Value'],
            ['Company Size', org.get('company_size', 'N/A')],
            ['Industry', org.get('industry', 'N/A')],
            ['Current Infrastructure', org.get('current_infrastructure', 'N/A')],
            ['IT Team Size', str(org.get('it_team_size', 'N/A'))],
            ['Cloud Experience', org.get('cloud_experience_level', 'N/A')],
        ]
        
        org_table = Table(org_data, colWidths=[2*inch, 3*inch])
        org_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(org_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Scoring methodology
        story.append(Paragraph("Scoring Methodology", self.styles['SubsectionHeader']))
        methodology = report.assessment_inputs.scoring_methodology
        
        if methodology.get('evaluation_criteria'):
            story.append(Paragraph("Evaluation Criteria:", self.styles['Normal']))
            for criterion in methodology['evaluation_criteria']:
                story.append(Paragraph(f"• {criterion}", self.styles['Normal']))
        
        return story
    
    def _build_appendices(self, report: ComprehensiveReport) -> List[Any]:
        """Build appendices section"""
        story = []
        
        story.append(Paragraph("Appendices", self.styles['SectionHeader']))
        
        # Cost breakdown
        if 'detailed_cost_breakdown' in report.appendices:
            story.append(Paragraph("Detailed Cost Breakdown", self.styles['SubsectionHeader']))
            
            cost_data = [['Provider', 'Compute', 'Storage', 'Network', 'Database', 'Other']]
            
            for provider, costs in report.appendices['detailed_cost_breakdown'].items():
                cost_data.append([
                    provider,
                    f"${costs['compute_cost']:,.0f}",
                    f"${costs['storage_cost']:,.0f}",
                    f"${costs['network_cost']:,.0f}",
                    f"${costs['database_cost']:,.0f}",
                    f"${costs['other_services']:,.0f}"
                ])
            
            cost_table = Table(cost_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fff3e0')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(cost_table)
        
        return story


def export_report_to_pdf(report: ComprehensiveReport) -> bytes:
    """
    Convenience function to export a report to PDF.
    
    Args:
        report: The comprehensive report to export
        
    Returns:
        bytes: PDF file content as bytes
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "ReportLab is required for PDF export. "
            "Install with: pip install reportlab"
        )
    
    service = PDFExportService()
    return service.export_to_pdf(report)