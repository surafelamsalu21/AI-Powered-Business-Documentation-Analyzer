from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
import datetime
from io import BytesIO
import base64


class ReportGenerator:
    def __init__(self, analysis_results, output_path=None):
        """
        Initializes the ReportGenerator with analysis results and output path.

        :param analysis_results: Dictionary containing TAM, SAM, SOM and related details.
        :param output_path: Path where the PDF report will be saved. If None, uses a buffer.
        """
        self.analysis_results = analysis_results
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.buffer = BytesIO() if not output_path else None

    def _add_title_page(self, doc_elements):
        """
        Adds a title page to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        title = "Market Analysis Report"
        subtitle = f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        doc_elements.append(Spacer(1, 2 * inch))
        doc_elements.append(Paragraph(title, self.styles['Title']))
        doc_elements.append(Spacer(1, 0.2 * inch))
        doc_elements.append(Paragraph(subtitle, self.styles['Normal']))
        doc_elements.append(PageBreak())

    def _add_summary(self, doc_elements):
        """
        Adds a summary section to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        summary_title = "Executive Summary"
        summary_content = (
            "This report provides a comprehensive analysis of the Total Addressable Market (TAM), "
            "Serviceable Available Market (SAM), and Serviceable Obtainable Market (SOM) based on "
            "the latest data extracted from business documents and external sources."
        )
        doc_elements.append(Paragraph(summary_title, self.styles['Heading1']))
        doc_elements.append(Spacer(1, 0.2 * inch))
        doc_elements.append(Paragraph(summary_content, self.styles['Justify']))
        doc_elements.append(PageBreak())

    def _add_tam_section(self, doc_elements):
        """
        Adds the TAM section to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        tam_title = "Total Addressable Market (TAM)"
        tam_value = f"${self.analysis_results.get('TAM', 'N/A'):,.2f}"
        tam_description = (
            "TAM represents the overall revenue opportunity available if your product "
            "or service achieves 100% market share in the specified market."
        )

        doc_elements.append(Paragraph(tam_title, self.styles['Heading1']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(
            Paragraph(f"<b>TAM Value:</b> {tam_value}", self.styles['Normal']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(Paragraph(tam_description, self.styles['Justify']))
        doc_elements.append(Spacer(1, 0.2 * inch))

        # Add data sources
        data_sources = self.analysis_results.get('data_sources', {})
        tam_source = data_sources.get(
            'TAM', 'Data derived from business documents and external APIs.')
        doc_elements.append(
            Paragraph("<b>Data Source:</b>", self.styles['Normal']))
        doc_elements.append(Paragraph(tam_source, self.styles['Normal']))
        doc_elements.append(PageBreak())

    def _add_sam_section(self, doc_elements):
        """
        Adds the SAM section to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        sam_title = "Serviceable Available Market (SAM)"
        sam_value = f"${self.analysis_results.get('SAM', 'N/A'):,.2f}"
        sam_description = (
            "SAM is the segment of the TAM targeted by your products and services which is "
            "within your operational reach."
        )

        doc_elements.append(Paragraph(sam_title, self.styles['Heading1']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(
            Paragraph(f"<b>SAM Value:</b> {sam_value}", self.styles['Normal']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(Paragraph(sam_description, self.styles['Justify']))
        doc_elements.append(Spacer(1, 0.2 * inch))

        # Add data sources
        data_sources = self.analysis_results.get('data_sources', {})
        sam_source = data_sources.get(
            'SAM', 'Data derived from business documents and external APIs.')
        doc_elements.append(
            Paragraph("<b>Data Source:</b>", self.styles['Normal']))
        doc_elements.append(Paragraph(sam_source, self.styles['Normal']))
        doc_elements.append(PageBreak())

    def _add_som_section(self, doc_elements):
        """
        Adds the SOM section to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        som_title = "Serviceable Obtainable Market (SOM)"
        som_value = f"${self.analysis_results.get('SOM', 'N/A'):,.2f}"
        som_description = (
            "SOM is the portion of SAM that you can capture realistically, considering your "
            "current and near-future capabilities."
        )

        doc_elements.append(Paragraph(som_title, self.styles['Heading1']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(
            Paragraph(f"<b>SOM Value:</b> {som_value}", self.styles['Normal']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(Paragraph(som_description, self.styles['Justify']))
        doc_elements.append(Spacer(1, 0.2 * inch))

        # Add data sources
        data_sources = self.analysis_results.get('data_sources', {})
        som_source = data_sources.get(
            'SOM', 'Data derived from business documents and external APIs.')
        doc_elements.append(
            Paragraph("<b>Data Source:</b>", self.styles['Normal']))
        doc_elements.append(Paragraph(som_source, self.styles['Normal']))
        doc_elements.append(PageBreak())

    def _add_market_insights(self, doc_elements):
        """
        Adds additional market insights to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        insights_title = "Market Insights"
        insights_content = (
            "Based on the analysis, the following insights have been derived:\n"
            "- The TAM indicates a significant opportunity within the market.\n"
            "- The SAM reflects a targeted segment that aligns with the company's strategic goals.\n"
            "- The SOM represents a realistic capture of the market, considering current resources and competition."
        )

        doc_elements.append(Paragraph(insights_title, self.styles['Heading1']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        for line in insights_content.split('\n'):
            doc_elements.append(Paragraph(line, self.styles['Normal']))
            doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(PageBreak())

    def _add_visualizations(self, doc_elements):
        """
        Adds visualizations (charts/graphs) to the PDF.

        :param doc_elements: List of PDF elements to append to.
        """
        import matplotlib.pyplot as plt

        # TAM, SAM, SOM Bar Chart
        tam = self.analysis_results.get('TAM', 0)
        sam = self.analysis_results.get('SAM', 0)
        som = self.analysis_results.get('SOM', 0)

        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = ['TAM', 'SAM', 'SOM']
        values = [tam, sam, som]
        colors_list = ['blue', 'green', 'red']
        ax.bar(metrics, values, color=colors_list)
        ax.set_title('Market Size Breakdown')
        ax.set_ylabel('USD')
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close(fig)
        buf.seek(0)
        img = buf.read()
        doc_elements.append(Spacer(1, 0.2 * inch))
        doc_elements.append(
            Paragraph("Market Size Breakdown", self.styles['Heading2']))
        doc_elements.append(Spacer(1, 0.1 * inch))
        doc_elements.append(Paragraph('<img src="{}" width="400" />'.format("data:image/png;base64," +
                                                                            base64.b64encode(img).decode()), self.styles['Normal']))
        doc_elements.append(Spacer(1, 0.2 * inch))
        buf.close()

        # Competitors Pie Chart
        competitors = self.analysis_results.get('competitors', [])
        if competitors:
            sizes = [100 / len(competitors)] * \
                len(competitors)  # Example distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(sizes, labels=competitors,
                   autopct='%1.1f%%', startangle=140, colors=colors_list[:len(competitors)])
            ax.axis('equal')
            ax.set_title('Market Share by Competitors')
            buf = BytesIO()
            plt.savefig(buf, format='PNG')
            plt.close(fig)
            buf.seek(0)
            img = buf.read()
            doc_elements.append(Spacer(1, 0.2 * inch))
            doc_elements.append(
                Paragraph("Market Share by Competitors", self.styles['Heading2']))
            doc_elements.append(Spacer(1, 0.1 * inch))
            doc_elements.append(Paragraph('<img src="{}" width="400" />'.format("data:image/png;base64," +
                                                                                base64.b64encode(img).decode()), self.styles['Normal']))
            doc_elements.append(Spacer(1, 0.2 * inch))
            buf.close()

    def generate_pdf(self, buffer=None):
        """
        Generates the PDF report and saves it to the specified output path or buffer.

        :param buffer: BytesIO buffer to write the PDF into. If None, writes to self.output_path.
        :return: Bytes of the generated PDF if buffer is provided; otherwise, None.
        """
        doc = SimpleDocTemplate(
            self.output_path if self.output_path else "temp_market_analysis_report.pdf",
            pagesize=LETTER,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        doc_elements = []

        # Add sections to the PDF
        self._add_title_page(doc_elements)
        self._add_summary(doc_elements)
        self._add_tam_section(doc_elements)
        self._add_sam_section(doc_elements)
        self._add_som_section(doc_elements)
        self._add_market_insights(doc_elements)
        self._add_visualizations(doc_elements)

        # Build the PDF
        doc.build(doc_elements)

        if buffer:
            with open(doc.filename, 'rb') as f:
                buffer.write(f.read())
            return buffer.getvalue()
        else:
            print(f"PDF report generated successfully at '{self.output_path}'")
            return None
