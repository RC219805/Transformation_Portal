Configuration Loader
====================

.. automodule:: transformation_portal.config_loader
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

YAML-based recipe and configuration loading with environment variable expansion
and path resolution.

Usage Example
-------------

.. code-block:: python

    from transformation_portal.config_loader import load_recipe, list_recipes

    # Load a recipe file
    recipe = load_recipe("/path/to/your/recipe.yaml")

    # Access recipe data (dict)
    print(recipe["name"])
    print(recipe.get("description", ""))
    print(recipe["stages"])

    # List available recipes
    recipes = list_recipes("/path/to/your/recipes")
    for recipe_info in recipes:
        print(recipe_info["name"], recipe_info["path"])

Supply an existing recipe file: ``config/recipes/luxury_estate.yaml`` is not a
checked-in fixture. ``list_recipes`` returns dictionaries (including an ``error``
field for invalid files), not ``Path`` objects. Path expansion is configuration
processing; strict filesystem authorization still belongs at the point of use.
