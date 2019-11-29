import logging
from concurrent.futures import ProcessPoolExecutor
from os import path, mkdir

from py21cmfast import yaml
from .cosmoHammer import (
    CosmoHammerSampler,
    LikelihoodComputationChain,
    HDFStorageUtil,
    Params,
)

logger = logging.getLogger("21cmFAST")


def build_computation_chain(core_modules, likelihood_modules, params=None, setup=True):
    """
    Build a likelihood computation chain from core and likelihood modules.

    Parameters
    ----------
    core_modules : list
        A list of objects which define the necessary methods to be core modules (see :mod:`~py21cmmc.mcmc.core`).
    likelihood_modules : list
        A list of objects which define the necessary methods to be likelihood modules (see
        :mod:`~py21cmmc.mcmc.likelihood`)
    params : :class:`~py21cmmc.mcmc.cosmoHammer.util.Params`, optional
        If provided, parameters which will be sampled by the chain.

    Returns
    -------
    chain : :class:`py21cmmc.mcmc.cosmoHammer.LikelihoodComputationChain.LikelihoodComputationChain`
    """
    if not hasattr(core_modules, "__len__"):
        core_modules = [core_modules]

    if not hasattr(likelihood_modules, "__len__"):
        likelihood_modules = [likelihood_modules]

    chain = LikelihoodComputationChain(params)

    for cm in core_modules:
        chain.addCoreModule(cm)

    for lk in likelihood_modules:
        chain.addLikelihoodModule(lk)

    if setup:
        chain.setup()
    return chain


def run_mcmc(
    core_modules,
    likelihood_modules,
    params,
    datadir=".",
    model_name="21CMMC",
    continue_sampling=True,
    reuse_burnin=True,
    log_level_21CMMC=None,
    **mcmc_options,
):
    """

    Parameters
    ----------
    core_modules : list
        A list of objects which define the necessary methods to be core modules (see :mod:`~py21cmmc.mcmc.core`).
    likelihood_modules : list
        A list of objects which define the necessary methods to be likelihood modules (see
        :mod:`~py21cmmc.mcmc.likelihood`)
    params : dict
        Parameters which will be sampled by the chain. Each entry's key specifies the name of the parameter, and
        its value is an iterable `(val, min, max, width)`, with `val` the initial guess, `min` and `max` the hard
        boundaries on the parameter's value, and `width` determining the size of the initial ball of walker positions
        for the parameter.
    datadir : str, optional
        Directory to which MCMC info will be written (eg. logs and chain files)
    model_name : str, optional
        Name of the model, which determines filenames of outputs.
    continue_sampling : bool, optional
        If an output chain file can be found that matches these inputs, sampling can be continued from its last
        iteration, up to the number of iterations specified. If set to `False`, any output file which matches these
        parameters will have its samples over-written.
    reuse_burnin : bool, optional
        If a pre-computed chain file is found, and `continue_sampling=False`, setting `reuse_burnin` will salvage
        the burnin part of the chain for re-use, but re-compute the samples themselves.
    log_level_mcmc : (int or str, optional)
        The logging level of the cosmoHammer log file.
        for details. This setting affects only the file output by the MCMC run.
    log_level_mcmc_stream : (int or str, optional)
        The logging level of the stdout/stderr stream from cosmoHammer. This has the same output as the cosmoHammer
        log file, but is printed to screen. See `log_level_mcmc` for input specifications.
    log_level_21CMMC : (int or str, optional)
        The logging level of the 21cmFAST Python code (specifically the "21CMMC" logging object). By default, this
        logger has only a stdout handler. See https://docs.python.org/3/library/logging.html#logging-levels


    Other Parameters
    ----------------
    All other parameters are passed directly to :class:`~py21cmmc.mcmc.cosmoHammer.CosmoHammerSampler.CosmoHammerSampler`.
    These include important options such as `walkersRatio` (the number of walkers is ``walkersRatio*nparams``),
    `sampleIterations`, `burninIterations`, `pool` and `threadCount`. It also contains `logLevel` and `log_level_stream`,
    which set the logging levels for the file and stream handlers of cosmoHammer respectively.

    Returns
    -------
    sampler : `~py21cmmc.mcmc.cosmoHammer.CosmoHammerSampler.CosmoHammerSampler` instance.
        The sampler object, from which the chain itself may be accessed (via the ``samples`` attribute).
    """
    file_prefix = path.join(datadir, model_name)

    try:
        mkdir(datadir)
    except FileExistsError:
        pass

    # Setup parameters.
    if not isinstance(params, Params):
        params = Params(*[(k, v) for k, v in params.items()])

    chain = build_computation_chain(
        core_modules, likelihood_modules, params, setup=False
    )

    if continue_sampling:
        try:
            with open(file_prefix + ".LCC.yml", "r") as f:
                old_chain = yaml.load(f)

            if old_chain != chain:
                raise RuntimeError(
                    "Attempting to continue chain, but chain parameters are different. "
                    + "Check your parameters against {file_prefix}.LCC.yml".format(
                        file_prefix=file_prefix
                    )
                )

        except FileNotFoundError:
            pass

        # We need to ensure that simulate=False if trying to continue sampling.
        for lk in chain.getLikelihoodModules():
            if hasattr(lk, "_simulate") and lk._simulate:
                logger.warning(
                    """
Likelihood {} was defined to re-simulate data/noise, but this is incompatible with
`continue_sampling`. Setting simulate=False and continuing...
"""
                )
                lk._simulate = False

    # Write out the parameters *before* setup.
    # TODO: not sure if this is the best idea -- should it be after setup()?
    try:
        with open(file_prefix + ".LCC.yml", "w") as f:
            yaml.dump(chain, f)
    except Exception as e:
        logger.warning(
            "Attempt to write out YAML file containing LikelihoodComputationChain failed. "
            "Boldly continuing..."
        )
        print(e)

    chain.setup()

    # Set logging levels
    if log_level_21CMMC is not None:
        logging.getLogger("21CMMC").setLevel(log_level_21CMMC)

    sampler = CosmoHammerSampler(
        continue_sampling=continue_sampling,
        likelihoodComputationChain=chain,
        storageUtil=HDFStorageUtil(file_prefix),
        filePrefix=file_prefix,
        reuseBurnin=reuse_burnin,
        pool=mcmc_options.get(
            "pool", ProcessPoolExecutor(max_workers=mcmc_options.get("threadCount", 1))
        ),
        **mcmc_options,
    )

    # The sampler writes to file, so no need to save anything ourselves.
    sampler.startSampling()

    return sampler


def run_multinest(
    core_modules,
    likelihood_modules,
    params,
    datadir=".",
    model_name="21CMMC",
    log_level_21CMMC=None,
    n_live_points=100,
    importance_nested_sampling=True,
    sampling_efficiency=0.8,
    evidence_tolerance=0.5,
    max_iter=50,
    multimodal=True,
    write_output=True,
    resume=True,
    **multinest_options,
):
    """

    Parameters
    ----------
    core_modules : list
        A list of objects which define the necessary methods to be core modules (see :mod:`~py21cmmc.mcmc.core`).
    likelihood_modules : list
        A list of objects which define the necessary methods to be likelihood modules (see
        :mod:`~py21cmmc.mcmc.likelihood`)
    params : dict
        Parameters which will be sampled by the chain. Each entry's key specifies the name of the parameter, and
        its value is an iterable `(val, min, max, width)`, with `val` the initial guess, `min` and `max` the hard
        boundaries on the parameter's value, and `width` determining the size of the initial ball of walker positions
        for the parameter.
    datadir : str, optional
        Directory to which MCMC info will be written (eg. logs and chain files)
    model_name : str, optional
        Name of the model, which determines filenames of outputs.
    log_level_mcmc : (int or str, optional)
        The logging level of the cosmoHammer log file.
        for details. This setting affects only the file output by the MCMC run.
    log_level_mcmc_stream : (int or str, optional)
        The logging level of the stdout/stderr stream from cosmoHammer. This has the same output as the cosmoHammer
        log file, but is printed to screen. See `log_level_mcmc` for input specifications.
    log_level_21CMMC : (int or str, optional)
        The logging level of the 21cmFAST Python code (specifically the "21CMMC" logging object). By default, this
        logger has only a stdout handler. See https://docs.python.org/3/library/logging.html#logging-levels

    Essential MultiNest Parameters
    ------------------------------
    n_live_points : int, optional
        number of live points
    importance_nested_sampling : bool, optional
        If True, Multinest will use Importance Nested Sampling (INS). Read http://arxiv.org/abs/1306.2144
        for more details on INS. Please read the MultiNest README file before using the INS in MultiNest v3.0.
    sampling_efficiency : float, optional
        defines the sampling efficiency. 0.8 and 0.3 are recommended for parameter estimation & evidence evalutation
    evidence_tolerance : float, optional
        A value of 0.5 should give good enough accuracy.
    max_iter : int, optional
        maximum number of iterations. 0 is unlimited.
    multimodal : bool, optional
        whether or not to detect multi mode
    write_output : bool, optional
        write output files? This is required for analysis.
    resume : bool, optional
        If True, Multinest will read in from file the last locations of the iso-likelihood contours to "continue"
        the sampling if the sampler hasn't sufficently converged.

    Other Parameters
    ----------------
    All other parameters are passed directly to multinest

    Returns
    -------
    sampler : multinest sampler
    """
    try:
        import pymultinest
    except ImportError:
        raise ImportError("You need to install pymultinest to use this function!")

    try:
        mkdir(datadir)
    except FileExistsError:
        pass

    try:
        mkdir(datadir + "/MultiNest")
    except FileExistsError:
        pass
    # Setup parameters.
    if not isinstance(params, Params):
        params = Params(*[(k, v) for k, v in params.items()])

    chain = build_computation_chain(
        core_modules, likelihood_modules, params, setup=False
    )

    # Write out the parameters *before* setup.
    # TODO: not sure if this is the best idea -- should it be after setup()?
    try:
        with open("%s/%s.LCC.yml" % (datadir, model_name), "w") as f:
            yaml.dump(chain, f)
    except Exception as e:
        logger.warning(
            "Attempt to write out YAML file containing LikelihoodComputationChain failed. "
            "Boldly continuing..."
        )
        print(e)

    chain.setup()

    # Set logging levels
    if log_level_21CMMC is not None:
        logging.getLogger("21CMMC").setLevel(log_level_21CMMC)

    def likelihood(p, ndim, nparams):
        input = [
            params[i][1] + p[i] * (params[i][2] - params[i][1]) for i in range(ndim)
        ]
        return chain.computeLikelihoods(
            chain.build_model_data(
                Params(*[(k, v) for k, v in zip(params.keys, input)])
            )
        )

    def prior(p, ndim, nparams):
        p = [params[i][1] + p[i] * (params[i][2] - params[i][1]) for i in range(ndim)]

    sampler = pymultinest.run(
        likelihood,
        prior,
        n_dims=len(params.keys),
        n_params=len(params.keys),
        n_live_points=n_live_points,
        resume=resume,
        write_output=write_output,
        outputfiles_basename=datadir + "/MultiNest/" + model_name,
        max_iter=max_iter,
        importance_nested_sampling=importance_nested_sampling,
        multimodal=multimodal,
        evidence_tolerance=evidence_tolerance,
        sampling_efficiency=sampling_efficiency,
        init_MPI=False,
    )

    return sampler
