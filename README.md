# 📄 AI-Powered Business Documentation Analyzer

## Table of Contents

- [📄 AI-Powered Business Documentation Analyzer](#-ai-powered-business-documentation-analyzer)
  - [Table of Contents](#table-of-contents)
  - [🔍 Overview](#-overview)
  - [🚀 Features](#-features)
  - [🛠️ Technology Stack](#️-technology-stack)
  - [📂 File Structure](#-file-structure)
  - [⚙️ Installation](#️-installation)
  - [🖥️ Usage](#️-usage)
  - [📈 How It Works](#-how-it-works)
    - [1. Document Upload \& Text Extraction](#1-document-upload--text-extraction)
    - [2. Key Metrics Extraction with OpenAI](#2-key-metrics-extraction-with-openai)
    - [3. Market Analysis via Perplexity AI](#3-market-analysis-via-perplexity-ai)
    - [4. TAM, SAM, SOM Calculation](#4-tam-sam-som-calculation)
    - [5. Visualization \& Reporting](#5-visualization--reporting)
  - [🔒 Security](#-security)
  - [🤝 Contributing](#-contributing)
  - [📝 License](#-license)
  - [📞 Contact](#-contact)

## 🔍 Overview

The **AI-Powered Business Documentation Analyzer** is a comprehensive tool designed to streamline the analysis of business documents. Whether you're dealing with PDFs, PowerPoint presentations, or Word documents, this application leverages advanced AI technologies to extract, analyze, and generate insightful reports, enabling businesses to make data-driven decisions with ease.

## 🚀 Features

-    **Document Upload & Processing**

     -    Supports multiple file formats: PDF, PPTX, DOCX.
     -    Extracts text content from uploaded documents seamlessly.

-    **Key Metrics Extraction**

     -    Utilizes OpenAI's GPT-4 to extract critical business metrics such as:
          -    **Market Size**
          -    **Target Audience**
          -    **Competitors**

-    **Market Analysis with Perplexity AI**

     -    Conducts in-depth market research by querying Perplexity AI.
     -    Generates insightful market data and trends.

-    **TAM, SAM, SOM Calculation**

     -    Automatically calculates:
          -    **Total Addressable Market (TAM)**
          -    **Serviceable Available Market (SAM)**
          -    **Serviceable Obtainable Market (SOM)**

-    **Executive Summary Generation**

     -    Crafts a detailed executive summary based on extracted metrics and market analysis.

-    **Visualization & Reporting**

     -    Generates visual charts to represent market data.
     -    Compiles all findings into a structured PDF report ready for download.

-    **User-Friendly Interface**
     -    Built with Streamlit for an intuitive and responsive user experience.

## 🛠️ Technology Stack

-    **Frontend & Deployment:**
     -    [Streamlit](https://streamlit.io/) - For building the interactive web application.
-    **Backend & Processing:**
     -    [Python](https://www.python.org/) - Core programming language.
     -    [PyPDF2](https://pypi.org/project/PyPDF2/) - PDF text extraction.
     -    [python-pptx](https://python-pptx.readthedocs.io/) - PowerPoint file processing.
     -    [python-docx](https://python-docx.readthedocs.io/) - Word document processing.
-    **AI & Machine Learning:**

     -    [OpenAI GPT-4](https://openai.com/) - For natural language processing and metric extraction.
     -    [Perplexity AI](https://www.perplexity.ai/) - For advanced market analysis and data fetching.

-    **Data Visualization:**
     -    [Matplotlib](https://matplotlib.org/) - Creating charts and graphs.
-    **PDF Generation:**
     -    [ReportLab](https://www.reportlab.com/) - For compiling PDF reports.
-    **Other Libraries:**
     -    [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis.
     -    [Requests](https://requests.readthedocs.io/) - Handling HTTP requests.
     -    [Logging](https://docs.python.org/3/library/logging.html) - Logging application events.

## 📂 File Structure

```
AI-Business-Analyzer/
├── src/
│   ├── main.py
│   ├── requirements.txt
│   └── utils.py
├── README.md
├── .gitignore
└── LICENSE
```

-    **src/main.py:** The core Streamlit application script.
-    **src/utils.py:** Helper functions and utility classes.
-    **requirements.txt:** List of dependencies required to run the application.
-    **README.md:** Project documentation.
-    **.gitignore:** Specifies intentionally untracked files to ignore.
-    **LICENSE:** Licensing information for the project.

## ⚙️ Installation

Follow these steps to set up the AI-Powered Business Documentation Analyzer on your local machine:

1. **Clone the Repository**

     ```bash
     git clone https://github.com/yourusername/AI-Business-Analyzer.git
     cd AI-Business-Analyzer
     ```

2. **Create a Virtual Environment**

     It's recommended to use a virtual environment to manage dependencies.

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

3. **Install Dependencies**

     ```bash
     pip install -r src/requirements.txt
     ```

4. **Set Up Environment Variables**

     Create a `.env` file in the `src/` directory and add your API keys:

     ```env
     OPENAI_API_KEY=your-openai-api-key
     PERPLEXITY_API_KEY=your-perplexity-api-key
     ```

     _Ensure you replace `your-openai-api-key` and `your-perplexity-api-key` with your actual API keys._

5. **Run the Application**

     ```bash
     streamlit run src/main.py
     ```

6. **Access the Application**

     Open your browser and navigate to `http://localhost:8501` to interact with the Analyzer.

## 🖥️ Usage

1. **Upload Documents**

     - Navigate to the application interface.
     - Click on the **"Upload your business documents"** button.
     - Select one or multiple files in PDF, PPTX, or DOCX formats.

2. **Process Documents**

     - Upon uploading, the application will extract text from the documents.
     - It will then analyze the extracted content to identify key metrics.

3. **View Extracted Metrics**

     - The application displays extracted metrics such as Market Size, Target Audience, and Competitors.

4. **Perform Market Analysis**

     - The tool queries Perplexity AI to gather additional market data.
     - It calculates TAM, SAM, and SOM based on the gathered data.

5. **Generate Reports**

     - A comprehensive analysis report is generated and displayed.
     - An executive summary synthesizing all insights is also provided.

6. **Download PDF Report**

     - Click on the **"📥 Download PDF Report"** button to obtain a detailed PDF version of your analysis.

## 📈 How It Works

### 1. Document Upload & Text Extraction

Users can upload business documents in supported formats. The application utilizes specialized libraries to extract raw text from these documents:

-    **PDFs:** Processed using `PyPDF2`.
-    **PowerPoint Presentations (PPTX):** Processed using `python-pptx`.
-    **Word Documents (DOCX):** Processed using `python-docx`.

### 2. Key Metrics Extraction with OpenAI

The extracted text is then fed into OpenAI's GPT-4 model to identify and extract essential business metrics:

-    **Market Size**
-    **Target Audience**
-    **Competitors**

This structured data forms the foundation for deeper analysis.

### 3. Market Analysis via Perplexity AI

To enrich the analysis, the tool engages with Perplexity AI, querying specific market-related questions based on the extracted metrics. This interaction fetches additional insights, trends, and data points relevant to the business context.

### 4. TAM, SAM, SOM Calculation

Using both the extracted metrics and data from Perplexity AI, the application calculates:

-    **Total Addressable Market (TAM):** The overall revenue opportunity available.
-    **Serviceable Available Market (SAM):** The segment of TAM targeted by the business.
-    **Serviceable Obtainable Market (SOM):** The portion of SAM the business can realistically capture.

### 5. Visualization & Reporting

With the calculated metrics, the tool generates visual charts for better comprehension. All findings, along with an executive summary, are compiled into a structured PDF report, providing a comprehensive overview of the business analysis.

## 🔒 Security

-    **API Keys Protection:** API keys for OpenAI and Perplexity AI are managed securely using environment variables. Ensure that your `.env` file is excluded from version control to prevent unauthorized access.
-    **Data Privacy:** Uploaded documents are processed in-memory and are not stored or transmitted to unauthorized third parties. Users are advised to handle sensitive documents responsibly.

## 🤝 Contributing

Contributions are welcome! If you'd like to enhance the AI-Powered Business Documentation Analyzer, please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

     ```bash
     git checkout -b feature/YourFeatureName
     ```

3. **Commit Your Changes**

     ```bash
     git commit -m "Add some feature"
     ```

4. **Push to the Branch**

     ```bash
     git push origin feature/YourFeatureName
     ```

5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For any questions, suggestions, or feedback, please reach out to:

-    **Email:** [surafelamsalu2013@gmail.com](mailto:your.email@example.com)
-    **GitHub:** [surafelasalu21](https://github.com/yourusername)

---

_Thank you for using the AI-Powered Business Documentation Analyzer! We hope it empowers your business insights and decision-making processes._
# AI-Powered-Business-Documentation-Analyzer
# AI-Powered-Business-Documentation-Analyzer
# AI-Powered-Business-Documentation-Analyzer
