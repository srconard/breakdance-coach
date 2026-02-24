# {{ title }} — 3D Analysis
{% if source_url %}

> **Source:** [{{ source_url }}]({{ source_url }})
{% endif %}

> **Backend:** {{ backend }} | **Generated:** {{ generated_date }}

---

{% for step in steps %}
## {{ step.label }}

![[{{ step.model_filename }}#autoplay]]

---

{% endfor %}

*Generated with Breakdance Coach 3D Analyzer*
