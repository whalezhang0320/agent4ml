from __future__ import annotations


class IdentityTranslator:
    """Identity 路径翻译器。Docker/E2B 用，虚拟路径直传无翻译。

    Docker bind mount 物理对齐（/mnt/agent4ml/user-data 挂载到容器同路径），
    E2B 沙箱内路径就是虚拟路径，不需翻译。
    """

    def translate_path(self, virtual_path: str) -> str:
        return virtual_path

    def translate_command(self, command: str) -> str:
        return command

    def mask_output(self, output: str) -> str:
        return output
