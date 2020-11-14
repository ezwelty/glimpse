{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

    {% block summary %}
    {% if classes %}
    .. rubric:: Classes
    .. autosummary::
    {% for item in classes %}
      {{ item }}
    {% endfor %}
    {% endif %}
    {% if functions %}
    .. rubric:: Functions
    .. autosummary::
    {% for item in functions %}
      {{ item }}
    {% endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    {% for item in classes %}
    .. autoclass:: {{ item }}
      :members:
    {% endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    {% for item in functions %}
    .. autofunction:: {{ item }}
    {% endfor %}
    {% endif %}
    {% endblock %}
