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
    recipe = load_recipe("config/recipes/luxury_estate.yaml")
    
    # Access recipe data (dict)
    print(recipe["name"])
    print(recipe.get("description", ""))
    print(recipe["stages"])
    
    # List available recipes
    recipes = list_recipes("config/recipes")
    for recipe_path in recipes:
        print(recipe_path.name)
