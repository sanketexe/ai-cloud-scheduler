import os
import re

directory = r"e:\clg\TS_AI_CLOUD_SCHEDULER\backend\app\services\migration_advisor\migration_advisor"

for filename in os.listdir(directory):
    if filename.endswith(".py"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. Replace MigrationProject and other advisor models imports
        # Case: from app.services.startup_migration.models import ( ... )
        # Using a regex to capture the block and deciding if it contains User or not.
        
        # Strategy: 
        # Replace `from app.services.startup_migration.models import` with `from .models import` 
        # BUT check if `User` is in the list. If so, split it.
        
        # Regex for multi-line import
        pattern = r"from backend\.app\.services\.startup_migration\.models import \((.*?)\)"
        
        def replace_import_block(match):
            imports_str = match.group(1)
            # Remove comments
            imports_str = re.sub(r'#.*', '', imports_str)
            imports = [i.strip() for i in imports_str.replace('\n', ',').split(',') if i.strip()]
            
            advisor_models = []
            other_models = []
            
            for imp in imports:
                imp_name = imp.split(' as ')[0].strip()
                if imp_name == 'User':
                    other_models.append(imp)
                else:
                    advisor_models.append(imp)
            
            res = ""
            if advisor_models:
                res += f"from .models import (\n    " + ",\n    ".join(advisor_models) + "\n)"
            
            if other_models:
                if res: res += "\n"
                res += f"from app.models.models import {', '.join(other_models)}"
            
            return res

        new_content = re.sub(pattern, replace_import_block, content, flags=re.DOTALL)
        
        # 2. Single line imports
        # pattern: from app.services.startup_migration.models import X, Y
        pattern_single = r"from backend\.app\.services\.startup_migration\.models import (.*)"
        
        def replace_single_line(match):
            imports_str = match.group(1)
            # handle comments potentially? assuming simpler case
            if '(' in imports_str: return match.group(0) # handled by block regex hopefully (files often formatted)
            
            imports = [i.strip().strip(',') for i in imports_str.split(',')]
            advisor_models = []
            other_models = []
            
            for imp in imports:
                imp_name = imp.split(' as ')[0].strip()
                if imp_name == 'User':
                    other_models.append(imp)
                elif imp_name == 'StartupMigrationProject' or imp_name == 'StartupMigrationPlan':
                    # If it explicitly asks for startup models, we should keep it pointing to startup_migration but use .models (relative?? no, absolute)
                    # But wait, startup_migration is sibling.
                    # We will assume they want advisor models if name matches
                    advisor_models.append(imp)
                else:
                    advisor_models.append(imp)
            
            res = ""
            if advisor_models:
                res += f"from .models import {', '.join(advisor_models)}"
            if other_models:
                if res: res += "\n"
                res += f"from app.models.models import {', '.join(other_models)}"
            return res

        # Apply single line replacement carefully (ignoring checks we already did)
        # Applying on remainder
        new_content = re.sub(pattern_single, replace_single_line, new_content)

        if content != new_content:
            print(f"Fixing imports in {filename}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
