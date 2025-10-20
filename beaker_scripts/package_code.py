import glob
import os
import time
import warnings
from subprocess import PIPE, Popen

import git
from beaker import Beaker

from mujoco_thor.mujoco_thor_constants import ABS_PATH_OF_TOP_LEVEL_MUJOCO_THOR_DIR


def bash_command(cmd: str, cwd: str | None = None):
    process = Popen(cmd, cwd=cwd, stdout=PIPE, shell=True, executable="/bin/bash")
    return process.communicate()[0].decode("utf-8")


def get_git_unignored_files(repo_path: str, include_submodules: bool):
    old_cwd = os.getcwd()
    os.chdir(repo_path)

    try:
        repo = git.Repo(repo_path)

        # Get untracked files
        untracked_files_cmd = "git status --short | grep '^?' | cut -d ' ' -f2-"
        untracked_files = set(bash_command(cmd=untracked_files_cmd, cwd=repo_path).splitlines())

        # Get tracked files
        tracked_files = set(repo.git.ls_files().splitlines())

        # Combine and sort the file lists
        all_files = sorted(tracked_files | untracked_files)
        all_files = [os.path.abspath(p) for p in all_files]

        # Get submodules
        submodule_paths = bash_command(
            cmd="git config --file .gitmodules --get-regexp path | awk '{ print $2 }'"
        ).strip()
        for submodule_path in submodule_paths.splitlines():
            submodule_path = os.path.abspath(submodule_path)
            if submodule_path in all_files:
                all_files.remove(submodule_path)

            if include_submodules:
                try:
                    all_files.extend(
                        get_git_unignored_files(
                            submodule_path, include_submodules=include_submodules
                        )
                    )
                except git.InvalidGitRepositoryError:
                    warnings.warn(
                        f"Submodule at path {submodule_path} has not been initialized!"
                        f" will not be included in dataset.",
                        stacklevel=2,
                    )

    finally:
        os.chdir(old_cwd)

    return sorted(list(set(all_files)))


def main(
    workspace: str | None = None,
    include_git: bool = True,
    include_submodules: bool = True,
    excluded_extensions=(".ipynb", "TestVHACD"),
):
    beaker_obj = (
        Beaker.from_env(default_workspace=workspace, timeout=60) if workspace else Beaker.from_env()
    )

    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    dataset_name = f"{beaker_obj.account.name}_code_{start_time_str}"

    os.chdir(ABS_PATH_OF_TOP_LEVEL_MUJOCO_THOR_DIR)
    beaker_obj.dataset.create(
        dataset_name,
        *[
            os.path.relpath(p, ABS_PATH_OF_TOP_LEVEL_MUJOCO_THOR_DIR)
            for p in get_git_unignored_files(
                ABS_PATH_OF_TOP_LEVEL_MUJOCO_THOR_DIR, include_submodules=include_submodules
            )
            if os.path.exists(p) and not any(p.endswith(ext) for ext in excluded_extensions)
        ],
        *(glob.glob(".git/**", recursive=True) if include_git else []),
        quiet=True,
    )

    dataset_name_full = f"{beaker_obj.account.name}/{dataset_name}"
    print(dataset_name_full)

    print(f"Code has size {beaker_obj.dataset.size(dataset_name_full) / 1000000:0.2f} MB.")


if __name__ == "__main__":
    main(workspace="robo-molmo", include_git=False, include_submodules=False)
