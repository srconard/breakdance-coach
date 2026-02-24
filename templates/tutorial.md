# {{ title }}
{% if source_url %}

> **Source:** [{{ source_url }}]({{ source_url }})
{% endif %}

---

{% for step in steps %}
## Step {{ step.step_number }}: {{ step.label }}
**{{ step.start_time }} - {{ step.end_time }}**

![[{{ step.gif_filename }}]]

{{ step.description }}

---

{% endfor %}

*Generated with Breakdance Tutorial GIF Generator*
