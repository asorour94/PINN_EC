# src/sumo_project_manager.py (Enhanced version with better error handling)

import os
from pathlib import Path
from dynamita.tool import create_temp_folder, extract_dll_from_project, extract_parameters_from_project

class SumoProjectManager:
    def __init__(self, archive_path: str):
        """
        Initialize SUMO project manager with enhanced path resolution.
        
        Args:
            archive_path: Path to the SUMO archive file (.sumo)
        """
        # Try multiple path resolution strategies
        resolved_path = self._resolve_archive_path(archive_path)
        
        if not os.path.isfile(resolved_path):
            self._print_diagnostic_info(archive_path)
            raise FileNotFoundError(f"SUMO archive not found: {resolved_path}")
            
        self.archive = resolved_path
        self.tmp_dir = create_temp_folder(".")
        print(f"✅ SUMO archive found: {self.archive}")

    def _resolve_archive_path(self, archive_path: str) -> str:
        """
        Try multiple strategies to resolve the archive path.
        
        Args:
            archive_path: Original path from config
            
        Returns:
            Resolved absolute path
        """
        # Strategy 1: Use path as-is
        if os.path.isfile(archive_path):
            return os.path.abspath(archive_path)
        
        # Strategy 2: Try relative to current working directory
        cwd_path = os.path.join(os.getcwd(), os.path.basename(archive_path))
        if os.path.isfile(cwd_path):
            return os.path.abspath(cwd_path)
        
        # Strategy 3: Try parent directory
        parent_dir = os.path.dirname(os.getcwd())
        parent_path = os.path.join(parent_dir, "My SUMO Simulation", os.path.basename(archive_path))
        if os.path.isfile(parent_path):
            return os.path.abspath(parent_path)
        
        # Strategy 4: Search in common locations
        search_paths = [
            "C:/Users/asamir94/Desktop/GBM_PC/My SUMO Simulation",
            "C:/Users/asamir94/Desktop/GBM_PC/My SUMO Simulation-PINN",
            "./",
            "../",
            "../My SUMO Simulation",
        ]
        
        filename = os.path.basename(archive_path)
        for search_dir in search_paths:
            candidate = os.path.join(search_dir, filename)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        
        # If all strategies fail, return original path for error message
        return archive_path

    def _print_diagnostic_info(self, original_path: str):
        """Print diagnostic information to help user locate the file."""
        print("🔍 SUMO Archive Search Diagnostic:")
        print(f"   Original path: {original_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Looking for file: {os.path.basename(original_path)}")
        
        # List files in current directory
        print("\n📁 Files in current directory:")
        try:
            for item in os.listdir("."):
                if item.endswith((".sumo", ".msumo")):
                    print(f"   ✅ {item}")
                elif os.path.isdir(item):
                    print(f"   📂 {item}/")
        except PermissionError:
            print("   ❌ Permission denied")
        
        # Check parent directory
        parent_dir = os.path.dirname(os.getcwd())
        print(f"\n📁 Files in parent directory ({parent_dir}):")
        try:
            for item in os.listdir(parent_dir):
                if item.endswith((".sumo", ".msumo")) or "SUMO" in item:
                    item_path = os.path.join(parent_dir, item)
                    if os.path.isfile(item_path):
                        print(f"   ✅ {item}")
                    elif os.path.isdir(item_path):
                        print(f"   📂 {item}/")
                        # Check for .sumo files in subdirectory
                        try:
                            for subitem in os.listdir(item_path):
                                if subitem.endswith((".sumo", ".msumo")):
                                    print(f"      ✅ {subitem}")
                        except PermissionError:
                            pass
        except PermissionError:
            print("   ❌ Permission denied")

    def extract(self) -> tuple[str, str]:
        """
        Extracts the model DLL and init script from the .sumo archive.

        Returns:
            dll_path: full path to the extracted DLL file
            init_script_path: full path to the extracted init.scs file
        """
        # 1) DLL: path in tmp_dir, then extract
        dll_target = os.path.join(self.tmp_dir, "sumoproject.dll")
        extract_dll_from_project(self.archive, dll_target)

        # 2) Init script: path in tmp_dir, then extract
        init_target = os.path.join(self.tmp_dir, "init.scs")
        extract_parameters_from_project(self.archive, self.tmp_dir, init_target)

        return dll_target, init_target
    
if __name__ == "__main__":
    # Quick manual test with enhanced error handling
    test_paths = [
        r"C:/Users/asamir94/Desktop/GBM_PC/My SUMO Simulation/MLE with energy(Thesis model).sumo",
        r"./MLE with energy(Thesis model).sumo",
        r"MLE with energy(Thesis model).sumo"
    ]
    
    for path in test_paths:
        try:
            print(f"\nTesting path: {path}")
            mgr = SumoProjectManager(path)
            dll, init = mgr.extract()
            print("✅ Success!")
            print("DLL extracted to:", dll)
            print("Init script extracted to:", init)
            break
        except FileNotFoundError as e:
            print(f"❌ Failed: {e}")
            continue