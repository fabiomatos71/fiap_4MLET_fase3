        static string ObterPeriodoMes(List<MesEmprestimo> pLista, DateTime? pDataRef)
        {
            var mes = pLista.Where(x => x.AnoMesReferencia == pDataRef.Value.Date.MesAno()).Single();
            if (pDataRef.Value.Day < mes.DiaDePagamentoDaFolha)
                return "antes_folha";
            else if (pDataRef.Value.Day == mes.DiaDePagamentoDaFolha)
                return "dia_folha";
            else
                return "apos_folha";
        }
        public class LogDto
        {
            public string usuario { get; set; }
            public string PeriodoDoMes { get; set; }
            public DateTime? DataHoraCriacao { get; set; }
            public int Dia { get; set; }
            public int Mes { get; set; }
            public int Ano { get; set; }
            public string casoDeUso { get; set; }
        }
        public static IEnumerable ConsultaGerarLogSistema(Session pSessao)
        {
            var exclusao = new List<string>()
                            {
                                "uc2004",
                                "uc2023site",
                                "uc0131site",
                                "uc0253site",
                                "uc2034site",
                                "uc0233site",
                                "uc2033site",
                                "uc2031site",
                                "uc2032site",
                                "uc0255site",
                                "uc3002site",
                                "ucmanterpermissoesdepapeisdousuario",
                                "ucmanterpapeiseseususuarios",
                                "uc2074site",
                                "ucmanterusuario",
                                "uc0026_01",
                                "uc0026_02",
                                "uc0026_03",
                                "uc0026_04",
                                "uc0026_05",
                                "uc0026_06",
                                "uc0026_07",
                                "uc0026_08",
                                "ucmanterpapelusuariosite",
                                "uc0044_03",
                                "uc0044_04",
                                "uc0044_05",
                                "uc0044_06",
                                "uc0044_07"
                            };

            var exclusao_usuarios = new List<string>() // Lista de usuários que foram exlcuidos da consulta na base de logs
            {
                "xxxxxx",
                "yyyyyyy",
                 "..."
            };
            var logs0 = (from x in new XPQuery<LogCasoDeUsoSistemaBeneficio>(pSessao)
                         where x.DataHoraCriacao.Value.Year >= 2018
                         where !exclusao_usuarios.Contains(x.Usuario.UserName)
                         orderby x.Usuario.UserName, x.DataHoraCriacao
                         select x).ToList();
            // usuários
            int num_usuario = 0;
            var usuarios = (from x in logs0
                           group x by x.Usuario.UserName into g
                           select new
                           {
                               Usuario = g.Key,
                               novo_nome = "usuario_" + num_usuario++.ToString("00")
                           }).ToList();

            var dias_pagamento_folha = (from x in new XPQuery<MesEmprestimo>(pSessao)
                                        where x.Plano.IdentificadorDoSistema == Plano.IDENTIFICADOR_PLANO_II
                                        where x.AnoMesReferencia.Value.Year >= 2018
                                        select x).ToList();

            var logs1 = (from x in logs0
                         orderby x.Usuario.UserName, x.DataHoraCriacao
                         select new LogDto
                         {
                             usuario = usuarios.Where(u => x.Usuario.UserName == u.Usuario).Single().novo_nome,
                             PeriodoDoMes = ObterPeriodoMes(dias_pagamento_folha, x.DataHoraCriacao),
                             DataHoraCriacao = x.DataHoraCriacao,
                             Dia = x.DataHoraCriacao.Value.Day,
                             Mes = x.DataHoraCriacao.Value.Month,
                             Ano = x.DataHoraCriacao.Value.Year,
                             casoDeUso = ObterUltimaParte(x.NomeClasseCasoDeUso)
                         }).Where(x => !exclusao.Contains(x.casoDeUso)).ToList();


            var logs2 = new List<LogDto>();
            LogDto log2_anterior = null;
            foreach (var log in logs1)
            {
                var log2 = new LogDto
                {
                    usuario = log.usuario,
                    PeriodoDoMes = log.PeriodoDoMes,
                    DataHoraCriacao = log.DataHoraCriacao,
                    Dia = log.Dia,
                    Mes = log.Mes,
                    Ano = log.Ano,
                    casoDeUso = log.casoDeUso
                };
                if ((log2_anterior == null) || 
                   (log2.usuario != log2_anterior.usuario || log2.casoDeUso != log2_anterior.casoDeUso || log2.Dia != log2_anterior.Dia || log2.Mes != log2_anterior.Mes || log2.Ano != log2_anterior.Ano))
                {
                    log2_anterior = log2;
                    logs2.Add(log2);
                }
            }
            return logs2;
        }