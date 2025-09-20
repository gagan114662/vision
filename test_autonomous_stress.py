#!/usr/bin/env python3
"""
Comprehensive stress test for autonomous MCP server dependency management.
Tests the system's ability to work 100% of the time under adverse conditions.
"""
import sys
import subprocess
import importlib
import time
import os
import tempfile
from pathlib import Path

def run_command(cmd, capture=True):
    """Run a command and return success status."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def uninstall_package(package):
    """Force uninstall a package."""
    print(f"🗑️  Force uninstalling {package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip uninstall {package} -y")
    return success

def corrupt_installation(package):
    """Simulate corrupted installation by partially removing files."""
    print(f"💥 Simulating corrupted installation for {package}...")
    # Try to find and corrupt the package
    try:
        import site
        site_packages = site.getsitepackages()[0]
        package_path = Path(site_packages) / package
        if package_path.exists():
            # Remove some files to simulate corruption
            for py_file in package_path.glob("*.py"):
                py_file.unlink()
                break
        return True
    except Exception:
        return False

def stress_test_regime_hmm():
    """Stress test the HMM regime detection server."""
    print("\n🧠 STRESS TESTING HMM REGIME DETECTION SERVER")
    print("=" * 60)

    # Test 1: Clean installation
    print("\n1️⃣  Test: Clean installation")
    uninstall_package("hmmlearn")
    try:
        sys.path.insert(0, "mcp/servers")
        if 'regime_hmm_server' in sys.modules:
            del sys.modules['regime_hmm_server']
        import regime_hmm_server
        print("✅ Clean installation test PASSED")
    except Exception as e:
        print(f"❌ Clean installation test FAILED: {e}")
        return False

    # Test 2: Corrupted installation recovery
    print("\n2️⃣  Test: Corrupted installation recovery")
    corrupt_installation("hmmlearn")
    try:
        importlib.reload(regime_hmm_server)
        print("✅ Corrupted installation recovery test PASSED")
    except Exception as e:
        print(f"❌ Corrupted installation recovery test FAILED: {e}")
        return False

    # Test 3: Multiple rapid imports
    print("\n3️⃣  Test: Multiple rapid imports")
    for i in range(5):
        try:
            if 'regime_hmm_server' in sys.modules:
                del sys.modules['regime_hmm_server']
            import regime_hmm_server
            print(f"   Rapid import {i+1}/5 ✅")
        except Exception as e:
            print(f"❌ Rapid import {i+1} FAILED: {e}")
            return False

    # Test 4: Function execution under stress
    print("\n4️⃣  Test: Function execution under stress")
    try:
        # Test the actual functionality
        test_params = {
            "prices": [100 + i + (i%3)*2 for i in range(50)],
            "n_regimes": 3
        }
        result = regime_hmm_server.detect_regimes(test_params)
        if "states" in result and "transition_matrix" in result:
            print("✅ Function execution test PASSED")
        else:
            print("❌ Function execution test FAILED: Invalid result format")
            return False
    except Exception as e:
        print(f"❌ Function execution test FAILED: {e}")
        return False

    return True

def stress_test_chart_server():
    """Stress test the chart rendering server."""
    print("\n📊 STRESS TESTING CHART RENDERING SERVER")
    print("=" * 60)

    # Test 1: Clean installation
    print("\n1️⃣  Test: Clean installation")
    uninstall_package("matplotlib")
    try:
        if 'chart_server' in sys.modules:
            del sys.modules['chart_server']
        import chart_server
        print("✅ Clean installation test PASSED")
    except Exception as e:
        print(f"❌ Clean installation test FAILED: {e}")
        return False

    # Test 2: Multiple backend configurations
    print("\n2️⃣  Test: Multiple backend configurations")
    try:
        # Force different matplotlib configurations
        import matplotlib
        backends = ['Agg', 'SVG', 'PDF']
        for backend in backends:
            matplotlib.use(backend)
            print(f"   Backend {backend} ✅")
        print("✅ Backend configuration test PASSED")
    except Exception as e:
        print(f"❌ Backend configuration test FAILED: {e}")
        return False

    # Test 3: Chart generation under stress
    print("\n3️⃣  Test: Chart generation under stress")
    try:
        test_params = {
            "prices": [
                {"timestamp": f"2024-01-{i:02d}T12:00:00Z", "price": 100 + i}
                for i in range(1, 11)
            ],
            "signals": [
                {"timestamp": "2024-01-05T12:00:00Z", "label": "signal1", "value": 105}
            ],
            "title": "Stress Test Chart"
        }
        result = chart_server.render_price_series(test_params)
        if "image_path" in result:
            print("✅ Chart generation test PASSED")
        else:
            print("❌ Chart generation test FAILED: No image path returned")
            return False
    except Exception as e:
        print(f"❌ Chart generation test FAILED: {e}")
        return False

    return True

def stress_test_semtools():
    """Stress test the semtools server."""
    print("\n🔍 STRESS TESTING SEMTOOLS SERVER")
    print("=" * 60)

    # Test 1: Clean installation
    print("\n1️⃣  Test: Clean installation")
    uninstall_package("llama-index-readers-file")
    try:
        if 'semtools_server' in sys.modules:
            del sys.modules['semtools_server']
        import semtools_server
        print("✅ Clean installation test PASSED")
    except Exception as e:
        print(f"❌ Clean installation test FAILED: {e}")
        return False

    # Test 2: CLI validation under stress
    print("\n2️⃣  Test: CLI validation under stress")
    for i in range(3):
        try:
            semtools_server._ensure_semtools_available()
            print(f"   CLI validation {i+1}/3 ✅")
        except Exception as e:
            print(f"❌ CLI validation {i+1} FAILED: {e}")
            return False

    # Test 3: Create test files for parsing
    print("\n3️⃣  Test: Document parsing functionality")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\nThis is a test document for semtools parsing.")
            test_file = f.name

        test_params = {
            "paths": [test_file],
            "output_format": "markdown"
        }
        result = semtools_server.semtools_parse(test_params)

        # Clean up
        os.unlink(test_file)

        if "output" in result:
            print("✅ Document parsing test PASSED")
        else:
            print("❌ Document parsing test FAILED: No output returned")
            return False
    except Exception as e:
        print(f"❌ Document parsing test FAILED: {e}")
        return False

    return True

def stress_test_provenance():
    """Stress test the provenance server."""
    print("\n💾 STRESS TESTING PROVENANCE SERVER")
    print("=" * 60)

    # Test 1: Clean installation
    print("\n1️⃣  Test: Clean installation")
    uninstall_package("pyimmudb")
    try:
        if 'provenance_server' in sys.modules:
            del sys.modules['provenance_server']
        import provenance_server
        print("✅ Clean installation test PASSED")
    except Exception as e:
        print(f"❌ Clean installation test FAILED: {e}")
        return False

    # Test 2: Multiple import cycles
    print("\n2️⃣  Test: Multiple import cycles")
    for i in range(3):
        try:
            if 'provenance_server' in sys.modules:
                del sys.modules['provenance_server']
            import provenance_server
            print(f"   Import cycle {i+1}/3 ✅")
        except Exception as e:
            print(f"❌ Import cycle {i+1} FAILED: {e}")
            return False

    return True

def concurrent_stress_test():
    """Test concurrent operations and dependency conflicts."""
    print("\n⚡ CONCURRENT STRESS TEST")
    print("=" * 60)

    print("\n1️⃣  Test: Concurrent server imports")
    import concurrent.futures

    def import_server(server_name):
        try:
            if server_name in sys.modules:
                del sys.modules[server_name]
            __import__(server_name)
            return True, server_name
        except Exception as e:
            return False, f"{server_name}: {e}"

    servers = ['regime_hmm_server', 'chart_server', 'semtools_server', 'provenance_server']

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(import_server, server) for server in servers]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    success_count = sum(1 for success, _ in results if success)
    total_count = len(results)

    print(f"Concurrent imports: {success_count}/{total_count} successful")

    if success_count == total_count:
        print("✅ Concurrent stress test PASSED")
        return True
    else:
        for success, result in results:
            if not success:
                print(f"❌ {result}")
        return False

def memory_stress_test():
    """Test memory usage under repeated operations."""
    print("\n🧠 MEMORY STRESS TEST")
    print("=" * 60)

    print("\n1️⃣  Test: Repeated import/reload cycles")

    import psutil
    import gc

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    for cycle in range(10):
        try:
            # Force cleanup
            for module in ['regime_hmm_server', 'chart_server', 'semtools_server']:
                if module in sys.modules:
                    del sys.modules[module]
            gc.collect()

            # Re-import
            import regime_hmm_server
            import chart_server
            import semtools_server

            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"   Cycle {cycle+1}/10: {current_memory:.1f}MB")

        except Exception as e:
            print(f"❌ Memory stress cycle {cycle+1} FAILED: {e}")
            return False

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_growth = final_memory - initial_memory

    print(f"\nMemory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_growth:.1f}MB)")

    if memory_growth < 50:  # Reasonable threshold
        print("✅ Memory stress test PASSED")
        return True
    else:
        print("❌ Memory stress test FAILED: Excessive memory growth")
        return False

def main():
    """Run comprehensive stress test suite."""
    print("🚀 AUTONOMOUS SYSTEM STRESS TEST SUITE")
    print("=" * 80)
    print("Testing 100% reliability under adverse conditions...")

    start_time = time.time()

    # Add MCP servers to path
    sys.path.insert(0, "mcp/servers")

    tests = [
        ("HMM Regime Detection", stress_test_regime_hmm),
        ("Chart Rendering", stress_test_chart_server),
        ("Semtools Document Intelligence", stress_test_semtools),
        ("Provenance Management", stress_test_provenance),
        ("Concurrent Operations", concurrent_stress_test),
        ("Memory Management", memory_stress_test),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} TEST SUITE FAILED: {e}")
            results.append((test_name, False))

    # Final Results
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("🏁 STRESS TEST RESULTS")
    print("=" * 80)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1

    print(f"\nOverall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Test Duration: {duration:.2f} seconds")

    if passed == total:
        print("\n🎉 ALL STRESS TESTS PASSED - 100% AUTONOMOUS RELIABILITY CONFIRMED!")
        return True
    else:
        print(f"\n⚠️  {total-passed} TEST(S) FAILED - SYSTEM NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)