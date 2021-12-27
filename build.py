from conans.client import conan_api
from cpt.packager import ConanMultiPackager
import os

def inspect_value_from_recipe(attribute, recipe_path):
    cwd = os.getcwd()
    result = None
    try:
        dir_name = os.path.dirname(recipe_path)
        os.chdir("./" if dir_name == "" else dir_name)
        conan_instance, _, _ = conan_api.Conan.factory()
        inspect_result = conan_instance.inspect(path=os.path.basename(recipe_path), attributes=[attribute])
        result = inspect_result.get(attribute)
    except:
        pass
    os.chdir(cwd)
    return result

def get_repo_branch_from_githubaction():
    def _clean_branch(branch):
        return branch[11:] if branch.startswith("refs/heads/") else branch
    repobranch = _clean_branch(os.getenv("GITHUB_REF", ""))
    if os.getenv("GITHUB_EVENT_NAME", "") == "pull_request":
        repobranch = os.getenv("GITHUB_BASE_REF", "")
    return repobranch

def has_shared_option(recipe_path):
    options = inspect_value_from_recipe(attribute="options", recipe_path=recipe_path)
    return options and "shared" in options

if __name__ == "__main__":
    username = "SpaceIm"
    recipe_path = os.path.abspath("conanfile.py")
    recipe_name = inspect_value_from_recipe(attribute="name", recipe_path=recipe_path)
    channel, recipe_version = get_repo_branch_from_githubaction().split("/")
    reference = "{}/{}@{}/{}".format(recipe_name, recipe_version, username, channel)
    shared_option_name = "{}:shared".format(recipe_name) if has_shared_option(recipe_path) else None

    builder = ConanMultiPackager(username=username, channel=channel,
                                 build_policy="missing", skip_check_credentials=True)
    builder.add_common_builds(shared_option_name=shared_option_name, pure_c=False, dll_with_static_runtime=True,
                              reference=reference, build_all_options_values=None)
    builder.run()
