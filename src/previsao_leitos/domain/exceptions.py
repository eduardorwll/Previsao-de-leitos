"""Hierarquia única de exceções do domínio Previsao-de-leitos."""


class PrevisaoLeitosError(Exception):
    """Exceção base para falhas explícitas de domínio, dados ou configuração."""


class ConfigurationError(PrevisaoLeitosError):
    """Erro em variáveis de ambiente, caminhos ou parâmetros de execução."""


class DataContractError(PrevisaoLeitosError):
    """Violação de contrato de schema, chaves, cardinalidade de join ou unidades físicas."""


class LeakageViolationError(PrevisaoLeitosError):
    """Tentativa de usar informação futura (ex-post) ou cruzar fronteira de split/holdout."""


class SourceUnavailableError(PrevisaoLeitosError):
    """Falha de conectividade, autenticação ou indisponibilidade em fonte externa."""
