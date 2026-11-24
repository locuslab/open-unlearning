When reviewing community contributions to OpenUnlearning:

- Check that documented commands select the intended trainer and run with the dependency versions pinned in this repository, including Transformers. Follow the command through Hydra config composition and trainer registration.
- Follow the layout of comparable merged contributions. Keep trainer implementations in `src/trainer/unlearn/` and trainer configs in `configs/trainer/`. Keep method-specific run scripts, sweep settings, and documentation in `community/methods/<method>/`.
- When a contribution changes the available methods or benchmarks, check the README inventory, counts, and affected documentation links.
- Use maintainer feedback on comparable merged PRs to judge documentation and cleanup requests. For each issue, identify the affected file, explain the consequence, and suggest the smallest fix.
