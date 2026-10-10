import subprocess
import tempfile
from pathlib import Path


def validate_lua(code, expected_event, expected_pattern):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.lua"
        path.write_text(code)

        # Run the generated Lua, then inspect registered autocommands.
        inspection = f"""
local autocmds = vim.api.nvim_get_autocmds({{
    event = "{expected_event}",
    pattern = "{expected_pattern}",
}})
if #autocmds == 0 then
    error("Expected autocommand not registered")
end
"""
        check_path = Path(tmp) / "check.lua"
        check_path.write_text(inspection)

        result = subprocess.run(
            [
                "nvim",
                "--headless",
                "-u",
                "NONE",
                "-i",
                "NONE",
                "-c",
                f"luafile {path}",
                "-c",
                f"luafile {check_path}",
                "-c",
                "qa!",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = result.stdout + result.stderr
        valid = (
            result.returncode == 0
            and "Error detected" not in output
            and "E5108:" not in output
            and "Expected autocommand not registered" not in output
        )
        return valid, output


if __name__ == "__main__":
    valid, error = validate_lua(
        """
    vim.api.nvim_create_autocmd("BufEnter", {
        pattern = "*.py",
        callback = function() end,
    })
    """,
        expected_event="BufEnter",
        expected_pattern="*.py",
    )

    print("Valid:", valid)
    print("Errors:", error)
