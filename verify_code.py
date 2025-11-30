import inspect
import database_manager

print("\n--- CHECKING ACTIVE CODE IN MEMORY ---")
source = inspect.getsource(database_manager._clean_properties)
print(source)
print("--------------------------------------\n")

if "dict(properties)" in source:
    print("✅ GREAT: Python sees the fix.")
else:
    print("❌ FATAL: Python is using the OLD code. The file update was ignored.")